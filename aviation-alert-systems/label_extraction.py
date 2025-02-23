import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import logging
from config import ModelConfig

logger = logging.getLogger(__name__)


class LabelExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = None
        self.bert_model = None
        self.config = ModelConfig()

    def setup_bert(self):
        """初始化BERT模型"""
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-chinese',
                do_lower_case=True,
                local_files_only=False
            )
            self.bert_model = BertModel.from_pretrained(
                'bert-base-chinese',
                local_files_only=False,
                output_hidden_states=False,
                output_attentions=False
            ).to(self.device)
            logger.info("BERT模型加载成功")
        except Exception as e:
            logger.warning(f"BERT模型加载失败: {e}")
            self.bert_tokenizer = None
            self.bert_model = None

    def extract_labels(self, remarks: pd.Series) -> np.ndarray:
        """标签提取主函数"""
        if self.bert_tokenizer is None or self.bert_model is None:
            return self._extract_labels_backup(remarks)

        try:
            labels = []
            for i in range(0, len(remarks), self.config.batch_size):
                batch_labels = self._process_batch(remarks[i:i + self.config.batch_size])
                labels.extend(batch_labels)

            return np.array(labels)

        except Exception as e:
            logger.error(f"BERT标签提取失败: {e}")
            return self._extract_labels_backup(remarks)

    def _process_batch(self, batch: pd.Series) -> list:
        """处理一个批次的数据"""
        batch = batch.tolist()
        inputs = self.bert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.config.bert_max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return [self._determine_label(remark) for remark in batch]

    def _determine_label(self, remark: str) -> int:
        """确定单个样本的标签"""
        if pd.isna(remark):
            return 1

        remark = str(remark).lower()
        if '假告警' in remark or '误报' in remark or '无需通报' in remark:
            return 0
        elif '真告警' in remark:
            return 1
        else:
            return 1

    def _extract_labels_backup(self, remarks: pd.Series) -> np.ndarray:
        """备用的标签提取方法"""
        return np.array([self._determine_label(remark) for remark in remarks])