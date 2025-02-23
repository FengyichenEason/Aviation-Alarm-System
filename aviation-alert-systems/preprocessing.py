import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import re
from typing import Dict, Tuple
import logging
from config import FeatureConfig
import gc

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.num_pattern = re.compile(r'\d+\.?\d*')
        self.feature_names = None  # 添加feature_names属性

    def preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的训练数据预处理流程"""
        try:
            data = df[FeatureConfig.NEEDED_COLUMNS].copy()

            # 内存优化
            self._optimize_memory(data)

            # 特征处理
            data = self._encode_categorical_features(data)
            data = self._create_numerical_features(data)
            data = self._create_temporal_features(data)
            data = self._create_network_features(data)

            # 设置特征名称
            self.feature_names = [
                '告警类型_encoded', '航班号三字码_encoded',
                '起飞机场_encoded', '目的机场_encoded',
                'alarm_value', 'alarm_hour', 'alarm_month', 'alarm_year',
                'dep_degree', 'arr_degree', 'route_importance'
            ]

            gc.collect()
            return data

        except Exception as e:
            logger.error(f"预处理失败: {e}")
            raise

    def preprocess_test_data(self, df: pd.DataFrame, feature_info: Dict) -> pd.DataFrame:
        """测试数据的最小预处理流程"""
        try:
            data = df[FeatureConfig.NEEDED_COLUMNS].copy()

            # 使用已有的编码器
            self.label_encoders = feature_info['label_encoders']
            self._apply_encoders(data)

            # 创建其他特征
            data = self._create_numerical_features(data)
            data = self._create_temporal_features(data)
            data = self._create_network_features(data)

            return data

        except Exception as e:
            logger.error(f"测试数据预处理失败: {e}")
            raise

    def _optimize_memory(self, df: pd.DataFrame) -> None:
        """内存优化"""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对分类特征进行编码"""
        for col in FeatureConfig.CATEGORICAL_FEATURES:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        return df

    def _apply_encoders(self, df: pd.DataFrame) -> None:
        """应用已有的编码器"""
        for col, encoder in self.label_encoders.items():
            df[f'{col}_encoded'] = df[col].map(
                dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            ).fillna(-1).astype(int)

    def _create_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建数值特征"""
        df['alarm_value'] = df['告警描述'].apply(self._extract_numbers)
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        df['alarm_hour'] = pd.to_datetime(df['告警时间']).dt.hour
        df['alarm_month'] = pd.to_datetime(df['告警日期']).dt.month
        df['alarm_year'] = pd.to_datetime(df['告警日期']).dt.year
        return df

    def _create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建网络特征"""
        try:
            G = nx.DiGraph()
            for _, row in df.iterrows():
                G.add_edge(row['起飞机场'], row['目的机场'])

            df['dep_degree'] = df['起飞机场'].map(dict(G.out_degree()))
            df['arr_degree'] = df['目的机场'].map(dict(G.in_degree()))
            df['route_importance'] = df.apply(
                lambda x: G.number_of_edges(x['起飞机场'], x['目的机场']), axis=1
            )

        except Exception as e:
            logger.error(f"创建网络特征失败: {e}")
            df['dep_degree'] = 0
            df['arr_degree'] = 0
            df['route_importance'] = 0

        return df

    def _extract_numbers(self, text: str) -> float:
        """从文本中提取数值"""
        try:
            if pd.isna(text):
                return -999
            numbers = self.num_pattern.findall(str(text))
            return float(numbers[0]) if numbers else -999
        except Exception as e:
            logger.error(f"提取数值失败: {e}")
            return -999