"""
src/inference/model_manager.py
===============================
模型管理器 — 负责模型文件的懒加载、健康检查、状态管理。

变更说明（vs 旧 app.py ModelManager 类）:
    - 旧: 耦合在 app.py 中，路径硬编码
    - 新: 独立为 inference 模块，路径通过 Settings 注入
    - 原因: 单一职责，可被 Flask 服务和 CLI 工具复用

特性:
    1. 懒加载: 首次调用 predict 时自动加载
    2. 状态管理: loaded / error / unloaded 三态
    3. 线程安全: 基于 threading.Lock 保证并发安全（如 Flask 多线程模式）
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.utils.exceptions import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器 — 统一管理模型、向量化器、标签转换器的加载与访问。

    状态机: UNLOADED → (load) → LOADED / ERROR

    Attributes:
        model: 分类器（OneVsRestClassifier 或兼容对象）
        vectorizer: TF-IDF 向量化器
        mlb: 标签二值化器（MultiLabelBinarizer）
        status: 当前状态 "unloaded" / "loaded" / "error"
        error_message: 加载失败时的错误信息

    Example:
        manager = ModelManager()
        manager.load()  # 或自动懒加载
        if manager.is_ready():
            labels = manager.get_labels()
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        mlb_path: Optional[Path] = None,
    ) -> None:
        """初始化模型管理器。

        Args:
            model_path: 模型文件路径（默认 settings.model_path）
            vectorizer_path: 向量化器路径（默认 settings.vectorizer_path）
            mlb_path: 标签转换器路径（默认 settings.mlb_path）
        """
        self.model_path = model_path or settings.model_path
        self.vectorizer_path = vectorizer_path or settings.vectorizer_path
        self.mlb_path = mlb_path or settings.mlb_path

        self.model: Any = None
        self.vectorizer: Any = None
        self.mlb: Any = None

        self.status: str = "unloaded"
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()

    # ============================================================
    # 加载 / 卸载
    # ============================================================

    def load(self) -> bool:
        """加载所有模型文件。

        线程安全: 使用锁防止并发加载。

        Returns:
            True 表示加载成功，False 表示失败

        Raises:
            ModelNotFoundError: 模型文件缺失
            ModelLoadError: 加载/反序列化失败
        """
        with self._lock:
            if self.status == "loaded":
                return True

            # 文件存在性检查
            for label, path in [
                ("模型", self.model_path),
                ("向量化器", self.vectorizer_path),
                ("标签转换器", self.mlb_path),
            ]:
                if not path.exists():
                    self.status = "error"
                    self.error_message = f"{label}文件不存在: {path}"
                    logger.error(self.error_message)
                    raise ModelNotFoundError(str(path))

            try:
                # 加载向量化器
                logger.info(f"加载向量化器: {self.vectorizer_path}")
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

                # 加载标签转换器
                logger.info(f"加载标签转换器: {self.mlb_path}")
                with open(self.mlb_path, "rb") as f:
                    self.mlb = pickle.load(f)

                # 加载模型
                logger.info(f"加载模型: {self.model_path}")
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)

                self.status = "loaded"
                self.error_message = None
                logger.info(
                    f"模型加载成功 — 标签数: {len(self.get_labels())}, "
                    f"标签: {self.get_labels()}"
                )
                return True

            except FileNotFoundError as e:
                self.status = "error"
                self.error_message = str(e)
                raise ModelNotFoundError(str(e)) from e
            except Exception as e:
                self.status = "error"
                self.error_message = f"加载失败: {e}"
                logger.error(self.error_message, exc_info=True)
                raise ModelLoadError(str(self.model_path), str(e)) from e

    def unload(self) -> None:
        """卸载模型以释放内存。"""
        with self._lock:
            self.model = None
            self.vectorizer = None
            self.mlb = None
            self.status = "unloaded"
            self.error_message = None
            logger.info("模型已卸载")

    # ============================================================
    # 状态查询
    # ============================================================

    def is_ready(self) -> bool:
        """模型是否已加载且可用。"""
        return self.status == "loaded" and self.model is not None

    def get_labels(self) -> List[str]:
        """获取模型支持的标签类别列表。

        Returns:
            标签名称列表（未加载时返回空列表）
        """
        if self.mlb is not None and hasattr(self.mlb, "classes_"):
            return list(self.mlb.classes_)
        return []

    @property
    def n_labels(self) -> int:
        """标签类别数。"""
        return len(self.get_labels())

    def health_check(self) -> Dict[str, Any]:
        """返回健康检查信息（供 API 使用）。

        Returns:
            包含状态、标签数、错误信息等的字典
        """
        return {
            "status": "ok" if self.is_ready() else "error",
            "model_loaded": self.is_ready(),
            "labels": self.get_labels(),
            "n_labels": self.n_labels,
            "error": self.error_message,
        }

    # ============================================================
    # 属性
    # ============================================================

    def __repr__(self) -> str:
        return f"ModelManager(status={self.status}, labels={self.n_labels})"
