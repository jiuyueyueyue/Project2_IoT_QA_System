"""
src/data_pipeline/cleaner.py
=============================
数据清洗器 — 解析标注 → 去重 → 过滤 → 保存清洗后数据。

变更说明（vs 旧 preprocess.py + train.py）:
    - 旧: parse_sentiment / deduplicate_data / filter_valid_data 零散函数
    - 新: 统一为 DataCleaner 类，管道化执行
    - 原因: 流水线封装，增加日志追踪每一步，便于调试

依赖:
    - pandas
    - src.utils.exceptions.DataCleanError
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.utils.exceptions import DataCleanError

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗流水线 — 解析标注、预处理文本、去重、过滤无效数据。

    Attributes:
        texts: 清洗后的文本列表
        labels: 清洗后的标签列表
        stats: 各步骤统计信息字典

    Example:
        cleaner = DataCleaner()
        texts, labels = cleaner.run(df)
        cleaner.save_cleaned_data()
    """

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.labels: List[List[str]] = []
        self.stats: dict = {
            "raw_count": 0,
            "after_parse": 0,
            "after_dedup": 0,
            "after_filter": 0,
        }

    # ============================================================
    # 标注解析
    # ============================================================

    @staticmethod
    def parse_sentiment(sentiment: object) -> List[str]:
        """解析 sentiment 字段中的 JSON/dict 获取标签列表。

        兼容格式:
            1. JSON 字符串: '{"choices": ["空调", "故障报错"]}'
            2. 双重转义 JSON（Label Studio 导出常见）: '"{\\"choices\\":[...]}"'
            3. Python dict 字面量字符串
            4. 已解析的 dict 对象

        Args:
            sentiment: sentiment 列原始值

        Returns:
            解析出的标签列表；解析失败返回空列表
        """
        # 1. None / NaN
        if sentiment is None or (isinstance(sentiment, float) and pd.isna(sentiment)):
            return []

        # 2. 已是 dict
        if isinstance(sentiment, dict):
            return sentiment.get("choices", []) or []

        # 3. 字符串解析
        if isinstance(sentiment, str):
            s = sentiment.strip()
            if not s:
                return []

            # 3a. 标准 JSON
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj.get("choices", []) or []
            except (json.JSONDecodeError, TypeError):
                pass

            # 3b. 双重转义 JSON（外层引号再去一层）
            try:
                obj = json.loads(json.loads(s))
                if isinstance(obj, dict):
                    return obj.get("choices", []) or []
            except (json.JSONDecodeError, TypeError):
                pass

            # 3c. Python dict 字面量（eval 最后兜底，仅用于可信数据源）
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return obj.get("choices", []) or []
            except (ValueError, SyntaxError):
                pass

        return []

    # ============================================================
    # 清洗流水线
    # ============================================================

    def run(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """执行完整数据清洗流水线。

        流程: 解析标注 → 基础预处理 → 去重 → 过滤无效数据

        Args:
            df: 原始 DataFrame（至少含 text, sentiment 列）
            save_path: 清洗后 CSV 保存路径（默认使用 settings.CLEANED_DATA_PATH）

        Returns:
            (文本列表, 标签列表)

        Raises:
            DataCleanError: 清洗过程中数据异常
        """
        self.stats["raw_count"] = len(df)
        logger.info(f"[清洗] 原始样本数: {len(df)}")

        # Step 1: 解析 sentiment → labels
        df = df.copy()
        df["labels"] = df["sentiment"].apply(self.parse_sentiment)

        # Step 2: 基础文本预处理（去标点/多余空格）
        from src.features.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        df["text_clean"] = df["text"].apply(
            lambda x: preprocessor.clean(x, remove_stopwords=False)
        )

        # Step 3: 过滤空标签 & 空文本
        df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
        df = df[df["text_clean"].apply(len) > 0].reset_index(drop=True)
        self.stats["after_parse"] = len(df)
        logger.info(f"[清洗] 过滤空标签/空文本后: {len(df)}")

        texts = df["text_clean"].tolist()
        labels = df["labels"].tolist()

        # Step 4: 去重
        texts, labels = self._deduplicate(texts, labels)
        self.stats["after_dedup"] = len(texts)
        logger.info(f"[清洗] 去重后: {len(texts)}")

        # Step 5: 过滤无效数据
        texts, labels = self._filter_valid(texts, labels)
        self.stats["after_filter"] = len(texts)
        logger.info(f"[清洗] 过滤无效数据后: {len(texts)}")

        self.texts = texts
        self.labels = labels

        # Step 6: 保存清洗后数据
        if save_path is not None:
            self.save_cleaned_data(save_path)
        else:
            self.save_cleaned_data()

        return texts, labels

    # ============================================================
    # 私有方法
    # ============================================================

    @staticmethod
    def _deduplicate(
        texts: List[str], labels: List[List[str]]
    ) -> Tuple[List[str], List[List[str]]]:
        """基于文本内容的去重（保留首次出现）。"""
        seen: set = set()
        unique_texts: List[str] = []
        unique_labels: List[List[str]] = []

        for text, label_list in zip(texts, labels):
            if text and text not in seen:
                seen.add(text)
                unique_texts.append(text)
                unique_labels.append(label_list)

        logger.info(f"  去重移除: {len(texts) - len(unique_texts)} 条")
        return unique_texts, unique_labels

    @staticmethod
    def _filter_valid(
        texts: List[str],
        labels: List[List[str]],
        min_text_length: int = 2,
    ) -> Tuple[List[str], List[List[str]]]:
        """过滤文本过短或无标签的无效样本。"""
        valid_texts: List[str] = []
        valid_labels: List[List[str]] = []

        for text, label_list in zip(texts, labels):
            if text and len(text) >= min_text_length and label_list and len(label_list) > 0:
                valid_texts.append(text)
                valid_labels.append(label_list)

        logger.info(f"  过滤移除: {len(texts) - len(valid_texts)} 条")
        return valid_texts, valid_labels

    # ============================================================
    # 数据持久化
    # ============================================================

    def save_cleaned_data(self, path: Optional[Path] = None) -> None:
        """保存清洗后的数据到 CSV 文件。

        Args:
            path: 保存路径，默认 settings.CLEANED_DATA_PATH
        """
        if path is None:
            path = settings.CLEANED_DATA_PATH

        path.parent.mkdir(parents=True, exist_ok=True)

        df_clean = pd.DataFrame({"text": self.texts, "labels": self.labels})
        df_clean.to_csv(path, index=False)
        logger.info(f"清洗后数据已保存: {path} ({len(df_clean)} 条)")

    # ============================================================
    # 统计摘要
    # ============================================================

    def summary(self) -> str:
        """返回清洗过程统计摘要。"""
        lines = [
            "=" * 50,
            "数据清洗摘要",
            "=" * 50,
            f"原始样本数:   {self.stats['raw_count']}",
            f"解析后样本数: {self.stats['after_parse']}",
            f"去重后样本数: {self.stats['after_dedup']}",
            f"最终样本数:   {self.stats['after_filter']}",
        ]
        if self.labels:
            # 统计标签分布
            label_counts: dict = {}
            for lbl_list in self.labels:
                for lbl in lbl_list:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            lines.append("-" * 50)
            lines.append("标签分布:")
            for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {lbl}: {cnt}")
        lines.append("=" * 50)
        return "\n".join(lines)
