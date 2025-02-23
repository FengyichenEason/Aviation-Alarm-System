import pandas as pd
import numpy as np
import os
import logging
import psutil
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict, List, Tuple
import joblib
from preprocessing import DataPreprocessor
from label_extraction import LabelExtractor
from model import AlarmClassifier
from evaluation import ModelEvaluator
from model_manager import ModelManager
from config import TrainingConfig

logger = logging.getLogger(__name__)


class AviationAlarmSystem:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.label_extractor = LabelExtractor()
        self.classifier = AlarmClassifier()
        self.evaluator = ModelEvaluator()
        self.model_manager = ModelManager()
        self.train_config = TrainingConfig()

    def train_pipeline(self, df: pd.DataFrame) -> Tuple:
        """完整的训练流程"""
        logger.info("开始训练流程...")
        try:
            # 初始化BERT
            self.label_extractor.setup_bert()

            # 数据预处理
            processed_data = self.preprocessor.preprocess_training_data(df)

            # 提取标签
            labels = self.label_extractor.extract_labels(processed_data['备注'])

            # 使用预处理器中的特征名称
            X = processed_data[self.preprocessor.feature_names].copy()

            # 重置索引以确保一致性
            X = X.reset_index(drop=True)
            labels = pd.Series(labels).reset_index(drop=True)
            processed_data = processed_data.reset_index(drop=True)

            # 训练测试集分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels,
                test_size=self.train_config.test_size,
                random_state=self.train_config.random_state,
                stratify=labels
            )

            # 训练模型
            models = self.classifier.train_with_confidence(X_train, y_train)

            # 评估模型
            predictions, probabilities = self.classifier.predict_with_ensemble(
                models, X_test
            )

            # 获取测试数据对应的原始数据
            test_indices = X_test.index
            test_data = processed_data.iloc[test_indices].copy()

            # 计算并打印详细的评估结果
            print("\n=== 训练评估结果 ===")
            error_analysis = self.evaluator.analyze_errors(
                y_test, predictions, test_data
            )
            self.evaluator.print_evaluation_results(
                y_test, predictions, error_analysis
            )

            # 打印特征重要性分析
            feature_importance = self.classifier.analyze_feature_importance(
                models, self.preprocessor.feature_names
            )
            print("\n=== 特征重要性分析 ===")
            print(feature_importance)

            # 保存结果
            self._save_training_results(
                models=models,
                features=self.preprocessor.feature_names,
                X_test=X_test,
                y_test=y_test,
                predictions=predictions,
                test_data=test_data
            )

            return models, X_test, y_test, predictions, test_data

        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            raise

    def test_pipeline(self, df: pd.DataFrame, model_path: str) -> Tuple:
        """测试流程"""
        logger.info("开始测试流程...")
        try:
            # 加载模型
            models, feature_info = self.model_manager.load_models(model_path)
            if models is None:
                raise ValueError("模型加载失败")

            # 数据预处理
            processed_data = self.preprocessor.preprocess_test_data(df, feature_info)

            # 提取特征
            X = processed_data[feature_info['feature_names']]

            # 如果有备注列，提取标签
            labels = None
            if '备注' in df.columns:
                labels = self.label_extractor.extract_labels(df['备注'])

            # 预测
            predictions, probabilities = self.classifier.predict_with_ensemble(models, X)

            # 评估
            if labels is not None:
                error_analysis = self.evaluator.analyze_errors(
                    labels, predictions, processed_data
                )
                self.evaluator.print_evaluation_results(
                    labels, predictions, error_analysis
                )

            return predictions, probabilities, processed_data, labels

        except Exception as e:
            logger.error(f"测试流程失败: {e}")
            raise

    def _save_training_results(
            self,
            models: List,
            features: List[str],
            X_test: pd.DataFrame,
            y_test: np.ndarray,
            predictions: np.ndarray,
            test_data: pd.DataFrame
    ) -> None:
        """保存训练结果"""
        try:
            # 计算评估指标
            error_analysis = self.evaluator.analyze_errors(
                y_test, predictions, test_data
            )

            # 准备保存的内容
            fold_scores = {
                'test_accuracy': (y_test == predictions).mean(),
                'error_analysis': error_analysis
            }

            # 准备特征信息
            feature_info = {
                'feature_names': features,
                'label_encoders': self.preprocessor.label_encoders
            }

            # 保存模型和相关信息
            save_path = self.model_manager.save_models(models, fold_scores)

            # 单独保存特征信息
            feature_file = os.path.join(save_path, 'feature_info.joblib')
            joblib.dump(feature_info, feature_file)
            logger.info(f"保存特征信息至 {feature_file}")

        except Exception as e:
            logger.error(f"保存训练结果失败: {e}")
            raise


def main():
    try:
        # 监控内存使用
        process = psutil.Process()
        logger.info(f"初始内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 初始化系统
        system = AviationAlarmSystem()

        # 加载数据
        logger.info("正在加载数据...")
        df = pd.read_excel('航空告警.xlsx', engine='openpyxl')
        logger.info(f"数据加载后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 显示数据时间跨度
        processed_dates = pd.to_datetime(df['告警日期'])
        start_date = processed_dates.min()
        end_date = processed_dates.max()
        logger.info(f"\n数据时间跨度: {start_date.year}年{start_date.month}月 至 {end_date.year}年{end_date.month}月")

        # 选择模式
        mode = input("请选择模式 (1: 训练新模型, 2: 加载已有模型测试): ").strip()

        if mode == "1":
            # 训练模式
            models, X_test, y_test, predictions, test_data = system.train_pipeline(df)

            # 输出训练结果
            system.evaluator.print_evaluation_results(
                y_test, predictions,
                system.evaluator.analyze_errors(y_test, predictions, test_data)
            )

            # 特征重要性分析
            feature_importance = system.classifier.analyze_feature_importance(
                models, system.preprocessor.feature_names
            )
            print("\n特征重要性分析:")
            print(feature_importance)

        elif mode == "2":
            # 测试模式
            model_versions = system.model_manager.list_saved_models()
            if not model_versions:
                logger.error("未找到已保存的模型")
                return

            print("\n可用的模型版本:")
            for i, version in enumerate(model_versions):
                info = system.model_manager.get_model_info(version)
                print(f"{i + 1}. {version}")
                print(f"   创建时间: {info['created_at']}")
                print(f"   模型数量: {info['num_folds']}")
                if 'test_accuracy' in info:
                    print(f"   测试集准确率: {info['test_accuracy']:.4f}")
                print()

            choice = int(input("\n请选择模型版本 (输入序号): ")) - 1
            if not (0 <= choice < len(model_versions)):
                logger.error("无效的选择")
                return

            model_path = os.path.join('saved_models', model_versions[choice])
            predictions, probabilities, processed_data, labels = system.test_pipeline(
                df, model_path
            )

        else:
            logger.error("无效的选择")

        logger.info(f"最终内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print("\n程序执行完成!")

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()