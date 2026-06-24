"""
src/inference/predictor.py
===========================
意图推理服务 — 基于已加载模型进行多标签文本意图识别。

业务定位:
    电商智能客服 NLU 意图引擎，对标京小智/飞鸽核心理解模块。
    输入用户消息 → 输出意图类别 + 置信度 + 路由策略，
    上游客服系统据此做分发（RAG 问答 / 订单 API / 转人工）。

特性:
    1. 概率阈值过滤（predict_proba 优先，predict 兜底）
    2. 自定义排序（按置信度降序）
    3. 意图路由匹配（INTENT_ROUTING: 意图 → 下游路由指令）
    4. 批量推理 + 单条异常隔离

变更说明（vs v2.0 IoT 版）:
    - solution_kb → intent_routing: 从维修方案改为路由分发策略
    - 返回字段 solution → action: 携带 route / target / priority
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from config.settings import FALLBACK_RESPONSE, INTENT_ROUTING
from src.inference.model_manager import ModelManager
from src.utils.exceptions import InferenceError

logger = logging.getLogger(__name__)


class Predictor:
    """电商意图识别推理器 — 输入用户消息，输出意图 + 路由策略。

    Attributes:
        model_manager: ModelManager 实例
        default_threshold: 默认置信度阈值
        default_max_labels: 默认最大返回意图数

    Example:
        manager = ModelManager()
        manager.load()
        predictor = Predictor(manager)
        results = predictor.predict("我的订单什么时候发货")
        # [{"label": "订单查询", "score": 0.85, "action": {"route": "order_api", ...}}]
    """

    def __init__(
        self,
        model_manager: ModelManager,
        default_threshold: float = 0.3,
        default_max_labels: int = 5,
        intent_routing: Optional[Dict[str, dict]] = None,
        fallback_response: Optional[str] = None,
    ) -> None:
        """初始化推理器。

        Args:
            model_manager: 已加载的模型管理器
            default_threshold: 默认置信度阈值（≥阈值的意图才返回）
            default_max_labels: 默认最大返回意图数
            intent_routing: 意图路由映射表（默认使用 config.INTENT_ROUTING）
            fallback_response: 兜底回复文本
        """
        self.model_manager = model_manager
        self.default_threshold = default_threshold
        self.default_max_labels = default_max_labels
        self.intent_routing = intent_routing or INTENT_ROUTING
        self.fallback_response = fallback_response or FALLBACK_RESPONSE

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
        """对单条用户消息进行意图识别。

        Args:
            text: 用户输入文本
            threshold: 置信度阈值（None 则使用默认值）
            max_labels: 最大返回意图数（None 则使用默认值）

        Returns:
            意图识别结果列表，每项包含:
                - label:   意图类别名称
                - score:   置信度 (0.0-1.0)
                - action:  路由指令 {route, target, priority, description}

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
                        route_info = self.intent_routing.get(label_name, {})
                        results.append({
                            "label": label_name,
                            "score": round(float(score), 4),
                            "action": {
                                "route": route_info.get("route", "fallback"),
                                "target": route_info.get("target", "兜底回复"),
                                "priority": route_info.get("priority", "normal"),
                                "description": route_info.get("description", ""),
                            },
                        })
            else:
                # 硬预测兜底
                logger.warning("模型不支持 predict_proba，使用 predict 硬预测")
                y_pred = model.predict(X_vec)
                for idx, name in enumerate(mlb.classes_):
                    if y_pred[0][idx] == 1:
                        route_info = self.intent_routing.get(str(name), {})
                        results.append({
                            "label": str(name),
                            "score": 1.0,
                            "action": {
                                "route": route_info.get("route", "fallback"),
                                "target": route_info.get("target", "兜底回复"),
                                "priority": route_info.get("priority", "normal"),
                                "description": route_info.get("description", ""),
                            },
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
        """批量意图识别。

        每条文本独立推理，单条失败不影响其他。

        Args:
            texts: 用户输入文本列表
            threshold: 置信度阈值
            max_labels: 最大返回意图数

        Returns:
            每条文本的意图识别结果列表
        """
        threshold = threshold if threshold is not None else self.default_threshold
        max_labels = max_labels if max_labels is not None else self.default_max_labels

        results: List[List[Dict]] = []
        for text in texts:
            try:
                results.append(self.predict(text, threshold, max_labels))
            except Exception as e:
                logger.error(f"批量推理单条失败: {e} (输入: {text[:30]}...)")
                results.append([])

        return results

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def build_fallback_result() -> List[Dict]:
        """构造兜底结果（当无意图命中时使用，建议转人工）。"""
        return [{
            "label": "意图未识别",
            "score": 0.0,
            "action": {
                "route": "human_agent",
                "target": "人工客服坐席",
                "priority": "normal",
                "description": "未识别到明确的业务意图，建议转人工客服处理",
            },
        }]

    @property
    def labels(self) -> List[str]:
        """获取模型支持的全部意图类别。"""
        return self.model_manager.get_labels()

    def get_routing_summary(self, results: List[Dict]) -> Dict:
        """从识别结果中提取路由决策摘要，供上层客服系统直接使用。

        Args:
            results: predict() 返回的意图列表

        Returns:
            {"primary_intent": str, "primary_route": str, "all_intents": [...], "suggest_action": str}
        """
        if not results:
            fallback = self.build_fallback_result()[0]
            return {
                "primary_intent": fallback["label"],
                "primary_route": "human_agent",
                "all_intents": [],
                "suggest_action": "转人工客服",
            }

        primary = results[0]
        return {
            "primary_intent": primary["label"],
            "primary_route": primary["action"]["route"],
            "all_intents": [r["label"] for r in results],
            "suggest_action": primary["action"]["target"],
        }
