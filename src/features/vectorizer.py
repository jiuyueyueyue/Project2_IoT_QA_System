"""
src/features/vectorizer.py
===========================
向量化管理器 — TF-IDF 特征提取与持久化。

变更说明（vs 旧 utils/train_eval.py train_vectorizer()）:
    - 旧: 独立函数，配置通过全局 TFIDF_CONFIG
    - 新: VectorizerManager 类封装，配置从 Settings 注入
    - 原因: 支持训练/推理两种使用场景，接口统一
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class VectorizerManager:
    """TF-IDF 特征提取管理器 — 负责向量化器的训练、保存、加载、转换。

    Attributes:
        vectorizer: sklearn TfidfVectorizer 实例
        is_fitted: 向量化器是否已训练

    Example:
        mgr = VectorizerManager({"max_features": 5000, "analyzer": "char"})
        mgr.fit(texts)
        X = mgr.transform(texts)
        mgr.save("./output/models/tfidf.pkl")
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """初始化向量化管理器。

        Args:
            config: TfidfVectorizer 参数字典（默认从 Settings.tfidf_config 读取）
        """
        if config is None:
            from config.settings import settings
            config = settings.tfidf_config

        logger.info(f"初始化 TF-IDF 向量化器，配置: {config}")
        self._config = config
        self.vectorizer = TfidfVectorizer(**config)
        self.is_fitted = False

    # ============================================================
    # 训练 / 转换
    # ============================================================

    def fit(self, texts: List[str]) -> "VectorizerManager":
        """在文本语料上训练向量化器。

        Args:
            texts: 文本列表

        Returns:
            self（支持链式调用）
        """
        logger.info(f"训练 TF-IDF 向量化器，语料大小: {len(texts)}")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"向量化器训练完成，特征维度: {self.n_features}")
        return self

    def transform(self, texts: List[str]):
        """将文本列表转换为 TF-IDF 稀疏特征矩阵。

        Args:
            texts: 文本列表

        Returns:
            scipy sparse matrix

        Raises:
            RuntimeError: 向量化器未训练
        """
        if not self.is_fitted:
            raise RuntimeError("向量化器未训练，请先调用 fit()")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """训练并一次性返回特征矩阵。"""
        self.fit(texts)
        return self.transform(texts)

    # ============================================================
    # 持久化
    # ============================================================

    def save(self, path: Path | str) -> None:
        """保存向量化器到 pkl 文件。

        Args:
            path: 保存路径

        Raises:
            RuntimeError: 向量化器未训练
        """
        if not self.is_fitted:
            raise RuntimeError("向量化器未训练，无法保存")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"向量化器已保存: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "VectorizerManager":
        """从 pkl 文件加载已训练的向量化器。

        Args:
            path: pkl 文件路径

        Returns:
            VectorizerManager 实例（已 fitted）

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"向量化器文件不存在: {path}")

        logger.info(f"加载向量化器: {path}")
        with open(path, "rb") as f:
            vectorizer = pickle.load(f)

        manager = cls(config={})  # 占位，下面直接赋值
        manager.vectorizer = vectorizer
        manager.is_fitted = True
        logger.info(f"向量化器加载完成，特征维度: {manager.n_features}")
        return manager

    # ============================================================
    # 属性
    # ============================================================

    @property
    def n_features(self) -> int:
        """特征维度数。"""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.get_feature_names_out())

    @property
    def config(self) -> Dict:
        """向量化器参数配置（只读）。"""
        return dict(self._config)
