from dataclasses import dataclass
import os
import logging
import ssl

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SSL和代理设置
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = '127.0.0.1:65534'
os.environ['HTTPS_PROXY'] = '127.0.0.1:65534'


@dataclass
class ModelConfig:
    """模型配置"""
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 200
    early_stopping_rounds: int = 10
    bert_max_length: int = 128
    batch_size: int = 32
    model_dir: str = 'saved_models'


@dataclass
class TrainingConfig:
    """训练配置"""
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5


class FeatureConfig:
    """特征配置"""
    NEEDED_COLUMNS = [
        '告警类型', '航班号三字码', '起飞机场', '目的机场',
        '告警描述', '告警时间', '告警日期', '备注'
    ]

    CATEGORICAL_FEATURES = [
        '告警类型', '航班号三字码', '起飞机场', '目的机场'
    ]