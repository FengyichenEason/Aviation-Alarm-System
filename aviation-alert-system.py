import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import jieba
import re
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import networkx as nx
import os
import ssl
import gc

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

# 设置环境变量
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = '127.0.0.1:65534'
os.environ['HTTPS_PROXY'] = '127.0.0.1:65534'

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AviationAlarmSystem:
    def __init__(self):
        self.device = get_device()
        print(f"使用设备: {self.device}")
        self.label_encoders = {}
        self.num_pattern = re.compile(r'\d+\.?\d*')

        # 设置较小的模型配置
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-chinese',
                do_lower_case=True,
                local_files_only=False
            )
            self.bert_model = BertModel.from_pretrained(
                'bert-base-chinese',
                local_files_only=False,
                output_hidden_states=False,  # 减少内存使用
                output_attentions=False  # 减少内存使用
            ).to(self.device)
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
            print("将使用备选方案进行文本处理")

    def preprocess_data(self, df):
        """数据预处理和特征工程"""
        """数据预处理和特征工程"""
        data = df.copy()

        # 只保留必要的列
        needed_columns = ['告警类型', '航班号三字码', '起飞机场', '目的机场',
                          '告警描述', '告警时间', '告警日期', '备注']
        data = data[needed_columns]

        # 使用类型转换节省内存
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category')

        # 分批处理特征编码
        for col in ['告警类型', '航班号三字码', '起飞机场', '目的机场']:
            self.label_encoders[col] = LabelEncoder()
            data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col])
            data[f'{col}_encoded'] = data[f'{col}_encoded'].astype('int32')

        # 提取数值特征
        data['alarm_value'] = data['告警描述'].apply(self._extract_numbers)
        data['alarm_value'] = data['alarm_value'].astype('float32')

        # 转换时间特征
        data['alarm_hour'] = pd.to_datetime(data['告警时间']).dt.hour.astype('int8')
        data['alarm_month'] = pd.to_datetime(data['告警日期']).dt.month.astype('int8')

        # 清理内存
        gc.collect()

        # 4. 创建机场网络特征

        self._create_airport_features(data)

        return data



    def _create_airport_features(self, data):
        """创建机场相关的高级特征"""
        # 构建机场网络
        G = nx.DiGraph()

        # 添加航线作为边
        for _, row in data.iterrows():
            G.add_edge(row['起飞机场'], row['目的机场'])

        # 计算机场中心性指标
        data['dep_degree'] = data['起飞机场'].map(dict(G.out_degree()))
        data['arr_degree'] = data['目的机场'].map(dict(G.in_degree()))
        data['route_importance'] = data.apply(
            lambda x: G.number_of_edges(x['起飞机场'], x['目的机场']), axis=1
        )

    def extract_labels(self, remarks):
        """使用BERT提取和推理标签"""
        if self.bert_tokenizer is None or self.bert_model is None:
            return self._extract_labels_backup(remarks)

        try:
            # 设置较小的批次大小
            batch_size = 32
            labels = []

            # 分批处理数据
            for i in range(0, len(remarks), batch_size):
                batch = remarks[i:i + batch_size].tolist()
                inputs = self.bert_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,  # 减小最大长度
                    return_tensors="pt"
                ).to(self.device)  # 移至GPU

                # 将输入数据移至GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # 先将张量移至CPU，然后转换为NumPy数组
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # 处理每个批次的结果
                for j, embedding in enumerate(batch_embeddings):
                    remark = remarks[i + j]
                    if pd.isna(remark):
                        labels.append(1)
                        continue

                    remark = str(remark).lower()
                    if '假告警' in remark or '误报' in remark or '无需通报' in remark:
                        labels.append(0)
                    elif '真告警' in remark:
                        labels.append(1)
                    # else:
                        # similarity_score = self._calculate_similarity_score(embedding)
                        # labels.append(1 if similarity_score > 0.1 else 0)
                    else:
                        # 默认标记为真实告警，确保不会漏掉潜在的重要告警
                        labels.append(1)

            return np.array(labels)

        except Exception as e:
            print(f"BERT处理失败: {e}")
            return self._extract_labels_backup(remarks)

    def _extract_labels_backup(self, remarks):
        """备用的标签提取方法"""
        labels = []
        for remark in remarks:
            if pd.isna(remark):
                labels.append(1)
                continue

            remark = str(remark).lower()
            if '假告警' in remark or '误报' in remark or '无需通报' in remark:
                labels.append(0)
            elif '真告警' in remark:
                labels.append(1)
            else:
                # 默认标记为真实告警，确保不会漏掉潜在的重要告警
                labels.append(1)

        return np.array(labels)

    def _calculate_similarity_score(self, embedding):
        """计算文本嵌入与已知标签样本的相似度"""
        # 这里可以实现更复杂的相似度计算逻辑
        return np.mean(embedding)

    def _extract_numbers(self, text):
        """从文本中提取数值"""
        try:
            if pd.isna(text):
                return -999
            numbers = self.num_pattern.findall(str(text))
            return float(numbers[0]) if numbers else -999
        except Exception as e:
            print(f"提取数值时出错: {e}")
            return -999

    def build_model(self, class_weights=None):
        """构建分层模型"""
        # 使用XGBoost处理类别不平衡
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            scale_pos_weight=class_weights,
            tree_method='gpu_hist',  # 更快的训练速度
            eval_metric='auc'
        )
        return model



    def train_with_confidence(self, X, y, confidence_threshold=0.8):
        """带置信度的训练过程"""
        skf = StratifiedKFold(n_splits=5)
        models = []

        # 创建模型保存目录
        import os
        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)

        # 计算类别权重
        class_weights = len(y[y == 0]) / len(y[y == 1])
        print(f"使用类别权重: {class_weights}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n训练折数 {fold + 1}/5")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.build_model(class_weights=len(y[y == 0]) / len(y[y == 1]))
            # 创建早停检验集
            eval_set = [(X_val, y_val)]
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=10)]
            )


            models.append(model)

            # 输出当前折的验证集性能
            val_pred = model.predict(X_val)
            print("验证集性能:")
            print(classification_report(y_val, val_pred))

        return models

    def predict_with_ensemble(self, models, X):
        """集成模型预测"""
        predictions = np.array([model.predict_proba(X) for model in models])
        # 使用投票机制
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred[:, 1] > 0.5, avg_pred[:, 1]

    def analyze_errors(self, y_true, y_pred, data):
        """错误分析"""
        # 确保所有输入长度一致
        assert len(y_true) == len(y_pred) == len(data), "输入长度不一致"

        # 创建错误样本掩码
        error_mask = y_true != y_pred
        errors = data[error_mask]

        # 创建对应长度的预测和真实值数组
        y_pred_errors = y_pred[error_mask]
        y_true_errors = y_true[error_mask]

        error_analysis = {
            'error_by_type': errors['告警类型'].value_counts(),
            'error_by_airport': errors.groupby(['起飞机场', '目的机场']).size(),
            'error_by_hour': errors['alarm_hour'].value_counts(),
            'false_positives': sum(y_pred_errors > y_true_errors),
            'false_negatives': sum(y_pred_errors < y_true_errors),
            'error_rate': len(errors) / len(y_true) * 100
        }

        return error_analysis


def main():
    try:
        # 导入内存监控
        import psutil
        process = psutil.Process()

        print("开始处理...")
        print(f"初始内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 加载数据
        print("正在加载数据...")
        df = pd.read_excel('航空告警.xlsx', engine='openpyxl')
        print(f"数据加载后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 创建系统实例
        print("初始化系统...")
        system = AviationAlarmSystem()

        # 数据预处理
        print("正在进行数据预处理...")
        processed_data = system.preprocess_data(df)
        print(f"预处理后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 提取标签
        print("正在提取标签...")
        labels = system.extract_labels(processed_data['备注'])
        print(f"标签提取后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 特征选择
        print("正在准备特征...")
        features = [
            '告警类型_encoded', '航班号三字码_encoded',
            '起飞机场_encoded', '目的机场_encoded',
            'alarm_value', 'alarm_hour', 'alarm_month',
            'dep_degree', 'arr_degree', 'route_importance'
        ]

        # 划分训练集和测试集
        print("划分训练集和测试集...")
        X = processed_data[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 输出数据集信息
        print("\n数据集信息:")
        print(f"总样本数: {len(labels)}")
        print(f"正样本数: {sum(labels == 1)}")
        print(f"负样本数: {sum(labels == 0)}")
        print(f"正负比例: {sum(labels == 1) / sum(labels == 0):.4f}")

        # 训练模型
        print("开始训练模型...")
        models = system.train_with_confidence(X_train, y_train)
        print("模型训练完成")

        # 预测和评估
        print("\n正在进行预测...")
        predictions, probabilities = system.predict_with_ensemble(models, X_test)

        # 输出评估报告
        print("\n分类报告:")
        print(classification_report(y_test, predictions, zero_division=1))

        # 错误分析
        print("\n正在进行错误分析...")
        # 确保使用相同索引的数据
        test_data = processed_data.loc[X_test.index]
        error_analysis = system.analyze_errors(y_test, predictions, test_data)

        print("\n错误分析结果:")
        for key, value in error_analysis.items():
            print(f"\n{key}:")
            print(value)

        # 添加混淆矩阵
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()