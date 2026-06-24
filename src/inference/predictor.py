"""
src/inference/predictor.py
===========================
推理服务 — 基于已加载模型进行多标签文本预测。

变更说明（vs 旧 app.py InferenceService 类）:
    - 旧: 耦合在 app.py 中，配置从 config 模块级读取
    - 新: 独立模块，参数可注入，支持单条/批量推理
    - 原因: 解耦后可在 Flask、CLI、Jupyter 等多场景复用

特性:
    1. 概率阈值过滤（predict_proba 优先，predict 兜底）
    2. 自定义排序（按置信度降序）
    3. 方案知识库匹配（SOLUTION_KB）
    4. 批量推理 + 单条异常隔离
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from config.settings import FALLBACK_SOLUTION, SOLUTION_KB
from src.inference.model_manager import ModelManager
from src.utils.exceptions import InferenceError

logger = logging.getLogger(__name__)


class Predictor:
    """多标签文本推理器。

    Attributes:
        model_manager: ModelManager 实例
        default_threshold: 默认置信度阈值
        default_max_labels: 默认最大返回标签数

    Example:
        manager = ModelManager()
        manager.load()
        predictor = Predictor(manager)
        results = predictor.predict("空调不制冷怎么办")
    """

    def __init__(
        self,
        model_manager: ModelManager,
        default_threshold: float = 0.3,
        default_max_labels: int = 10,
        solution_kb: Optional[Dict[str, str]] = None,
        fallback_solution: Optional[str] = None,
    ) -> None:
        """初始化推理器。

        Args:
            model_manager: 已加载的模型管理器
            default_threshold: 默认置信度阈值
            default_max_labels: 默认最大返回标签数
            solution_kb: 故障方案知识库（默认 SOLUTION_KB）
            fallback_solution: 兜底方案文本
        """
        self.model_manager = model_manager
        self.default_threshold = default_threshold
        self.default_max_labels = default_max_labels
        self.solution_kb = solution_kb or SOLUTION_KB
        self.fallback_solution = fallback_solution or FALLBACK_SOLUTION

        # 输入校验
        if not 0.0 <= default_threshold <= 1.0:
            logger.warning(f"阈值 {default_threshold} 不在 [0,1] 范围内，使用默认值 0.3")
            self.default_threshold = 0.3

    # ============================================================
    # 单条推理
    # ============================================================

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
        max_labels: Optional[int] = None,
    ) -> List[Dict]:
        """对单条文本进行多标签推理。

        Args:
            text: 输入文本
            threshold: 置信度阈值（None 则使用默认值）
            max_labels: 最大返回标签数（None 则使用默认值）

        Returns:
            预测结果列表，每项包含 label / score / solution

        Raises:
            InferenceError: 模型未加载或推理失败
        """
        if not self.model_manager.is_ready():
            raise InferenceError("模型未加载")

        threshold = threshold if threshold is not None else self.default_threshold
        max_labels = max_labels if max_labels is not None else self.default_max_labels

        try:
            model = self.model_manager.model
            vectorizer = self.model_manager.vectorizer
            mlb = self.model_manager.mlb

            # 特征转换
            X_vec = vectorizer.transform([text])

            # 推理
            results: List[Dict] = []

            if hasattr(model, "predict_proba"):
                # 概率预测模式（推荐）
                proba = model.predict_proba(X_vec)
                for idx, score in enumerate(proba[0]):
                    if score >= threshold:
                        label_name = str(mlb.classes_[idx])
                        results.append({
                            "label": label_name,
                            "score": round(float(score), 4),
                            "solution": self.solution_kb.get(label_name, self.fallback_solution),
                        })
            else:
                # 硬预测兜底
                logger.warning("模型不支持 predict_proba，使用 predict 硬预测")
                y_pred = model.predict(X_vec)
                for idx, name in enumerate(mlb.classes_):
                    if y_pred[0][idx] == 1:
                        results.append({
                            "label": str(name),
                            "score": 1.0,
                            "solution": self.solution_kb.get(str(name), self.fallback_solution),
                        })

            # 按置信度降序排列 & 截断
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:max_labels]

        except Exception as e:
            raise InferenceError(str(e), text=text) from e

    # ============================================================
    # 批量推理
    # ============================================================

    def batch_predict(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        max_labels: Optional[int] = None,
    ) -> List[List[Dict]]:
        """批量多标签推理。

        每条文本独立推理，单条失败不影响其他。

        Args:
            texts: 输入文本列表
            threshold: 置信度阈值
            max_labels: 最大返回标签数

        Returns:
            每条文本的预测结果列表
        """
        threshold = threshold if threshold is not None else self.default_threshold
        max_labels = max_labels if max_labels is not None else self.default_max_labels

        results: List[List[Dict]] = []
        for text in texts:
            try:
                results.append(self.predict(text, threshold, max_labels))
            except Exception as e:
                logger.error(f"批量推理单条失败: {e} (输入: {text[:30]}...)")
                results.append([])  # 失败返回空列表，不中断整体

        return results

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def build_fallback_result() -> List[Dict]:
        """构造兜底结果（当无标签命中时使用）。"""
        return [{
            "label": "未识别",
            "score": 0.0,
            "solution": FALLBACK_SOLUTION,
        }]

    @property
    def labels(self) -> List[str]:
        """获取模型支持的标签列表。"""
        return self.model_manager.get_labels()
