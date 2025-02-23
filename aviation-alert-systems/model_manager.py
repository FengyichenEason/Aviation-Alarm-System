import os
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from config import ModelConfig

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config = ModelConfig()
        self._create_model_directory()

    def _create_model_directory(self) -> None:
        """创建模型保存目录"""
        try:
            if not os.path.exists(self.config.model_dir):
                os.makedirs(self.config.model_dir)
                logger.info(f"创建模型保存目录: {self.config.model_dir}")

        except Exception as e:
            logger.error(f"创建模型目录失败: {e}")
            raise

    def save_models(self, models: List, fold_scores: Dict) -> str:
        """保存训练好的模型和评估分数"""
        try:
            # 生成模型版本标识
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.config.model_dir, f'model_v{timestamp}')
            os.makedirs(model_path, exist_ok=True)

            # 保存模型文件
            self._save_model_files(models, model_path)

            # 保存评估分数
            self._save_evaluation_scores(fold_scores, model_path)

            # 保存配置信息
            self._save_model_config(model_path, len(models))

            logger.info(f"模型及相关信息已保存至: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

    def _save_model_files(self, models: List, model_path: str) -> None:
        """保存模型文件"""
        for i, model in enumerate(models, 1):
            model_file = os.path.join(model_path, f'fold_{i}.joblib')
            joblib.dump(model, model_file)
            logger.info(f"保存模型 fold_{i} 至 {model_file}")

    def _save_evaluation_scores(self, fold_scores: Dict, model_path: str) -> None:
        """保存评估分数"""
        score_file = os.path.join(model_path, 'scores.joblib')
        joblib.dump(fold_scores, score_file)
        logger.info(f"保存评估分数至 {score_file}")

    def _save_model_config(self, model_path: str, num_models: int) -> None:
        """保存模型配置"""
        config_info = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': os.path.basename(model_path),
            'num_models': num_models,
            'hyperparameters': {
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'n_estimators': self.config.n_estimators
            }
        }

        config_file = os.path.join(model_path, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)

    def load_models(self, model_path: str) -> Tuple[Optional[List], Optional[Dict]]:
        """加载模型和特征信息"""
        try:
            # 加载特征信息
            feature_info = self._load_feature_info(model_path)

            # 加载模型文件
            models = self._load_model_files(model_path)

            if not models:
                raise ValueError("未找到任何模型文件")

            logger.info(f"成功加载模型，共{len(models)}个fold")
            return models, feature_info

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None, None

    def _load_feature_info(self, model_path: str) -> Dict:
        """加载特征信息"""
        feature_file = os.path.join(model_path, 'feature_info.joblib')
        if not os.path.exists(feature_file):
            logger.error(f"特征信息文件不存在: {feature_file}")
            raise FileNotFoundError(f"特征信息文件不存在: {feature_file}")

        try:
            feature_info = joblib.load(feature_file)
            self._validate_feature_info(feature_info)
            return feature_info
        except Exception as e:
            logger.error(f"加载特征信息失败: {e}")
            raise

    def _load_model_files(self, model_path: str) -> List:
        """加载模型文件"""
        models = []
        i = 1
        while True:
            model_file = os.path.join(model_path, f'fold_{i}.joblib')
            if not os.path.exists(model_file):
                break
            models.append(joblib.load(model_file))
            i += 1
        return models

    def _validate_feature_info(self, feature_info: Dict) -> None:
        """验证特征信息完整性"""
        required_keys = ['label_encoders', 'feature_names']
        missing_keys = [key for key in required_keys if key not in feature_info]
        if missing_keys:
            msg = f"特征信息不完整，缺少以下键: {missing_keys}"
            logger.error(msg)
            raise ValueError(msg)

    def list_saved_models(self) -> List[str]:
        """列出所有已保存的模型版本"""
        if not os.path.exists(self.config.model_dir):
            logger.warning(f"模型目录 {self.config.model_dir} 不存在")
            return []

        model_versions = [
            d for d in os.listdir(self.config.model_dir)
            if os.path.isdir(os.path.join(self.config.model_dir, d))
               and d.startswith('model_v')
        ]
        return sorted(model_versions, reverse=True)

    def get_model_info(self, model_version: str) -> Dict:
        """获取模型详细信息"""
        try:
            model_path = os.path.join(self.config.model_dir, model_version)

            # 获取配置和评分信息
            try:
                config_info = self._load_model_config(model_path)
            except FileNotFoundError:
                config_info = {}

            try:
                scores = self._load_model_scores(model_path)
            except FileNotFoundError:
                scores = {}

            # 统计模型文件数量
            model_files = [f for f in os.listdir(model_path) if f.startswith('fold_')]

            return {
                'version': model_version,
                'created_at': config_info.get('created_at', 'Unknown'),
                'num_folds': len(model_files),
                'test_accuracy': scores.get('test_accuracy', 'N/A'),
                'hyperparameters': config_info.get('hyperparameters', {}),
                'error_analysis': scores.get('error_analysis', {})
            }

        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {'version': model_version, 'error': str(e)}

    def _load_model_config(self, model_path: str) -> Dict:
        """加载模型配置信息"""
        config_file = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_file):
            logger.warning(f"配置文件不存在: {config_file}")
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_model_scores(self, model_path: str) -> Dict:
        """加载模型评分信息"""
        score_file = os.path.join(model_path, 'scores.joblib')
        if not os.path.exists(score_file):
            logger.warning(f"评分文件不存在: {score_file}")
            raise FileNotFoundError(f"评分文件不存在: {score_file}")

        return joblib.load(score_file)