import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import logging
from typing import List, Tuple, Dict, Optional
import gc
from config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class AlarmClassifier:
    def __init__(self):
        self.config = ModelConfig()
        self.train_config = TrainingConfig()

    def train_with_confidence(self, X: pd.DataFrame, y: np.ndarray) -> List[xgb.XGBClassifier]:
        """交叉验证训练"""
        try:
            logger.info("开始交叉验证训练...")
            # 确保 X 是 DataFrame, y 是 numpy array 或 Series
            X = X.reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True)

            skf = StratifiedKFold(n_splits=self.train_config.n_splits)
            models = []

            # 计算类别权重
            class_weights = len(y[y == 0]) / len(y[y == 1])
            logger.info(f"使用类别权重: {class_weights}")

            # 使用 numpy array 进行分割
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.Series) else y

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), 1):
                model = self._train_fold(
                    X, y, train_idx, val_idx,
                    class_weights, fold
                )
                if model is not None:
                    models.append(model)

            if not models:
                raise ValueError("所有折数训练都失败了")

            logger.info(f"训练完成，成功训练了 {len(models)} 个模型")
            return models

        except Exception as e:
            logger.error(f"训练过程失败: {e}")
            raise

    def _train_fold(self, X: pd.DataFrame, y: pd.Series,
                    train_idx: np.ndarray, val_idx: np.ndarray,
                    class_weights: float, fold: int) -> Optional[xgb.XGBClassifier]:
        """训练单个fold"""
        try:
            logger.info(f"\n训练折数 {fold}/{self.train_config.n_splits}")

            # 使用 iloc 按位置索引获取数据
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # 构建模型
            model = self._build_model(class_weights)
            eval_set = [(X_val, y_val)]

            # 训练
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_rounds
                )]
            )

            # 评估
            val_pred = model.predict(X_val)
            logger.info(f"\n第 {fold} 折验证集性能:")
            logger.info("\n" + classification_report(y_val, val_pred))

            gc.collect()
            return model

        except Exception as e:
            logger.error(f"第 {fold} 折训练失败: {e}")
            return None

    def _build_model(self, class_weights: float) -> xgb.XGBClassifier:
        """构建XGBoost模型"""
        return xgb.XGBClassifier(
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            scale_pos_weight=class_weights,
            tree_method='gpu_hist',
            eval_metric='auc'
        )

    def predict_with_ensemble(self, models: List[xgb.XGBClassifier],
                              X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """集成模型预测"""
        predictions = np.array([model.predict_proba(X) for model in models])
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred[:, 1] > 0.5, avg_pred[:, 1]

    def analyze_feature_importance(self, models: List[xgb.XGBClassifier],
                                   feature_names: List[str]) -> pd.DataFrame:
        """分析特征重要性"""
        importance_scores = np.zeros(len(feature_names))
        for model in models:
            importance_scores += model.feature_importances_

        importance_scores /= len(models)
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)