"""
app.py
======
电商智能客服 NLU 意图识别引擎 — Flask API 层

业务定位:
    对标京东京小智、抖音飞鸽智能客服的 NLU 意图识别模块。
    输入用户消息 → 输出意图类别 + 置信度 + 路由策略，
    上层客服系统据此做分发（RAG 问答 / 订单 API / 转人工）。

重构说明（v3.0 电商版）:
    - IoT 智能家居 → 电商客服意图识别（6 类业务意图）
    - 返回字段 solution → action（路由分发指令）
    - 其余架构沿用 v2.0 企业级分层

API 接口:
    GET    /                 渲染意图识别测试页
    GET    /api/health       健康检查 + 支持的意图列表
    GET    /api/version      API 版本信息
    POST   /api/predict      单条意图识别 {"text": "...", "threshold": 0.3}
    POST   /api/batch        批量意图识别 {"texts": [...], "threshold": 0.3}

启动方式:
    python main.py serve
"""

from __future__ import annotations

import re
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

from config.settings import settings, FALLBACK_RESPONSE
from src.inference.model_manager import ModelManager
from src.inference.predictor import Predictor
from src.utils.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from src.utils.logger import setup_logger

# ============================================================
# 日志
# ============================================================
logger = setup_logger("iot-qa-api")

# ============================================================
# 模型初始化（应用启动时加载）
# ============================================================
model_manager = ModelManager()

try:
    model_manager.load()
    logger.info("模型管理器初始化成功")
except (ModelNotFoundError, ModelLoadError) as e:
    logger.error(f"模型加载失败: {e}")
    # 继续启动，API 会返回 503 并提示用户

inference_service = Predictor(
    model_manager=model_manager,
    default_threshold=settings.INFERENCE_THRESHOLD,
    default_max_labels=settings.INFERENCE_MAX_LABELS,
)

# ============================================================
# 输入校验器
# ============================================================
class InputValidator:
    """API 层输入校验 — 用户消息长度/内容过滤、阈值校验。"""

    MAX_TEXT_LENGTH: int = 500
    MIN_TEXT_LENGTH: int = 2

    @classmethod
    def validate(cls, text: str) -> Dict[str, object]:
        """校验输入文本。

        Args:
            text: 用户输入字符串

        Returns:
            {"valid": bool, "message": str}
        """
        if not isinstance(text, str):
            return {"valid": False, "message": "输入必须为字符串"}

        text = text.strip()

        if len(text) < cls.MIN_TEXT_LENGTH:
            return {
                "valid": False,
                "message": f"输入文本过短，最少需要 {cls.MIN_TEXT_LENGTH} 个字符",
            }

        if len(text) > cls.MAX_TEXT_LENGTH:
            return {
                "valid": False,
                "message": f"输入文本过长，最多允许 {cls.MAX_TEXT_LENGTH} 个字符",
            }

        return {"valid": True, "message": "校验通过"}

    @staticmethod
    def sanitize(text: str) -> str:
        """规范化输入文本：去首尾空、合并空格。"""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def validate_threshold(value: object) -> float:
        """校验并转换阈值参数。

        Args:
            value: 原始 threshold 参数值

        Returns:
            浮点阈值

        Raises:
            ValueError: 阈值无效
        """
        threshold = float(value)  # type: ignore[arg-type]
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("阈值必须在 0.0-1.0 之间")
        return threshold


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__)
API_VERSION: str = "v3.0.0"


# ============================================================
# 页面路由
# ============================================================

@app.route("/")
def index() -> str:
    """渲染问答首页。"""
    return render_template("index.html")


@app.route("/api/version")
def version():
    """API 版本信息。"""
    return jsonify({
        "version": API_VERSION,
        "service": "电商智能客服 NLU 意图识别引擎",
    })


@app.route("/api/health")
def health():
    """健康检查 — 返回模型加载状态、支持的意图类别、路由策略等。"""
    health_info = model_manager.health_check()
    health_info.update({
        "version": API_VERSION,
        "threshold": settings.INFERENCE_THRESHOLD,
        "intent_routing": {
            label: info["route"]
            for label, info in settings.INTENT_ROUTING.items()
        },
    })
    return jsonify(health_info)


# ============================================================
# 预测 API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    """单条用户消息意图识别。

    Request Body:
        {"text": "我的订单什么时候发货", "threshold": 0.3}

    Response:
        {"code": 0, "msg": "ok", "data": [{"label": "...", "score": 0.85, "action": {...}}], "threshold": 0.3}
    """
    try:
        data: dict = request.get_json(silent=True) or {}

        if not isinstance(data, dict):
            return jsonify({"code": 400, "msg": "请求体必须为 JSON 对象", "data": []}), 400

        # 提取参数
        text: str = data.get("text", "")
        if not text:
            return jsonify({"code": 400, "msg": "请提供 text 参数", "data": []}), 400

        # 阈值校验
        try:
            threshold = InputValidator.validate_threshold(
                data.get("threshold", settings.INFERENCE_THRESHOLD)
            )
        except (ValueError, TypeError) as e:
            return jsonify({"code": 400, "msg": f"阈值参数无效: {e}", "data": []}), 400

        # 输入校验
        validation = InputValidator.validate(text)
        if not validation["valid"]:
            return jsonify({"code": 400, "msg": validation["message"], "data": []}), 400

        text = InputValidator.sanitize(text)

        # 模型状态检查
        if not model_manager.is_ready():
            return jsonify({
                "code": 503,
                "msg": "模型未加载，请先运行 python main.py train",
                "data": [],
            }), 503

        # 推理
        logger.debug(f"推理请求: text='{text[:50]}...', threshold={threshold}")
        results = inference_service.predict(text, threshold=threshold)

        if not results:
            results = Predictor.build_fallback_result()

        logger.info(f"推理完成: text='{text[:30]}...', labels={[r['label'] for r in results]}")

        return jsonify({
            "code": 0,
            "msg": "ok",
            "data": results,
            "threshold": threshold,
        })

    except InferenceError as e:
        logger.error(f"推理错误: {e}")
        return jsonify({"code": 500, "msg": str(e), "data": []}), 500
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        return jsonify({"code": 500, "msg": f"服务器内部错误: {str(e)}", "data": []}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """批量用户消息意图识别。

    Request Body:
        {"texts": ["我要退货", "快递到哪了"], "threshold": 0.3}

    Response:
        {"code": 0, "msg": "ok", "data": [[...], [...]], "total": 2, "valid": 2}
    """
    try:
        data: dict = request.get_json(silent=True) or {}

        if not isinstance(data, dict):
            return jsonify({"code": 400, "msg": "请求体必须为 JSON 对象", "data": []}), 400

        texts: list = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"code": 400, "msg": "请提供 texts 数组", "data": []}), 400

        if len(texts) > settings.INFERENCE_BATCH_SIZE:
            return jsonify({
                "code": 400,
                "msg": f"单次批量请求最多 {settings.INFERENCE_BATCH_SIZE} 条",
                "data": [],
            }), 400

        # 阈值校验
        try:
            threshold = InputValidator.validate_threshold(
                data.get("threshold", settings.INFERENCE_THRESHOLD)
            )
        except (ValueError, TypeError) as e:
            return jsonify({"code": 400, "msg": f"阈值参数无效: {e}", "data": []}), 400

        if not model_manager.is_ready():
            return jsonify({
                "code": 503,
                "msg": "模型未加载，请先运行 python main.py train",
                "data": [],
            }), 503

        # 逐条校验 & 清洗
        sanitized_texts: List[str] = []
        valid_indices: List[int] = []
        for idx, text in enumerate(texts):
            validation = InputValidator.validate(text)
            if validation["valid"]:
                sanitized_texts.append(InputValidator.sanitize(text))
                valid_indices.append(idx)
            else:
                logger.warning(f"批量请求第 {idx} 条校验失败: {validation['message']}")

        logger.debug(f"批量推理请求: {len(sanitized_texts)} 条有效")
        results = inference_service.batch_predict(sanitized_texts, threshold=threshold)

        # 重建输出（校验失败项填充错误）
        final_results: List[List[Dict]] = []
        result_idx = 0
        for idx in range(len(texts)):
            if idx in valid_indices:
                item = results[result_idx] if result_idx < len(results) else []
                final_results.append(item)
                result_idx += 1
            else:
                final_results.append([{
                    "label": "校验失败",
                    "score": 0.0,
                    "action": {
                        "route": "fallback",
                        "target": "无",
                        "priority": "low",
                        "description": "输入文本不符合要求",
                    },
                }])

        logger.info(f"批量推理完成: {len(texts)} 条")

        return jsonify({
            "code": 0,
            "msg": "ok",
            "data": final_results,
            "threshold": threshold,
            "total": len(texts),
            "valid": len(valid_indices),
        })

    except InferenceError as e:
        logger.error(f"批量推理错误: {e}")
        return jsonify({"code": 500, "msg": str(e), "data": []}), 500
    except Exception as e:
        logger.error(f"批量推理未知错误: {e}", exc_info=True)
        return jsonify({"code": 500, "msg": f"服务器内部错误: {str(e)}", "data": []}), 500


# ============================================================
# 错误处理器
# ============================================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"code": 400, "msg": "请求参数错误", "data": []}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"code": 404, "msg": "接口不存在", "data": []}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"code": 500, "msg": "服务器内部错误", "data": []}), 500


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    logger.info(f"启动 Flask 服务: {settings.FLASK_HOST}:{settings.FLASK_PORT}")
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG,
    )
