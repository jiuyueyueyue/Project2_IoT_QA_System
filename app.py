"""
app.py
======
Project2_IoT_QA_System - IoT 智能家居多标签故障问答系统 Flask 后端

功能特性：
1. 输入校验（长度限制、内容过滤）
2. 异常捕获（模型加载异常、推理异常、网络异常）
3. 阈值过滤（可配置置信度阈值）
4. 分级日志（DEBUG/INFO/WARNING/ERROR）
5. API 版本控制
6. 批量预测支持

启动方式：
    python app.py

打开浏览器访问：
    http://127.0.0.1:5000/

API 接口：
    GET    /              渲染问答首页
    GET    /api/health    健康检查，返回模型加载状态和服务信息
    GET    /api/version   获取 API 版本信息
    POST   /api/predict   单条预测，接收 JSON {"text": "故障描述", "threshold": 0.5}
    POST   /api/batch     批量预测，接收 JSON {"texts": [...], "threshold": 0.5}
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from config import (
    FALLBACK_SOLUTION,
    FLASK_CONFIG,
    INFERENCE_CONFIG,
    LOG_CONFIG,
    MODEL_DIR,
    SOLUTION_KB,
)

logger = logging.getLogger("iot-qa-api")
logger.setLevel(getattr(logging, LOG_CONFIG["level"]))

console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_CONFIG["level"]))
formatter = logging.Formatter(LOG_CONFIG["format"], datefmt=LOG_CONFIG["datefmt"])
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ModelManager:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.mlb = None
        self.loaded = False
        self.load_error = None

    def load(self):
        try:
            model_path = os.path.join(MODEL_DIR, "best_model_model.pkl")
            vectorizer_path = os.path.join(MODEL_DIR, "best_model_vectorizer.pkl")
            mlb_path = os.path.join(MODEL_DIR, "best_model_mlb.pkl")

            logger.info(f"加载模型: {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            logger.info(f"加载向量化器: {vectorizer_path}")
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

            logger.info(f"加载标签转换器: {mlb_path}")
            with open(mlb_path, "rb") as f:
                self.mlb = pickle.load(f)

            self.loaded = True
            logger.info("模型加载成功")

        except FileNotFoundError as e:
            self.load_error = f"模型文件缺失: {e}"
            logger.error(self.load_error)
        except Exception as e:
            self.load_error = f"模型加载失败: {e}"
            logger.error(self.load_error, exc_info=True)

    def is_ready(self) -> bool:
        return self.loaded and self.model is not None

    def get_labels(self) -> List[str]:
        if self.mlb is not None:
            return list(self.mlb.classes_)
        return []


model_manager = ModelManager()
model_manager.load()


class InputValidator:
    MAX_TEXT_LENGTH = 500
    MIN_TEXT_LENGTH = 2

    @staticmethod
    def validate(text: str) -> Dict[str, str]:
        if not isinstance(text, str):
            return {"valid": False, "message": "输入必须为字符串"}

        text = text.strip()

        if len(text) < InputValidator.MIN_TEXT_LENGTH:
            return {
                "valid": False,
                "message": f"输入文本过短，最少需要 {InputValidator.MIN_TEXT_LENGTH} 个字符",
            }

        if len(text) > InputValidator.MAX_TEXT_LENGTH:
            return {
                "valid": False,
                "message": f"输入文本过长，最多允许 {InputValidator.MAX_TEXT_LENGTH} 个字符",
            }

        return {"valid": True, "message": "校验通过"}

    @staticmethod
    def sanitize(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text


class InferenceService:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def predict(
        self,
        text: str,
        threshold: float = INFERENCE_CONFIG["threshold"],
        max_labels: int = INFERENCE_CONFIG["max_labels"],
    ) -> List[dict]:
        if not self.model_manager.is_ready():
            raise RuntimeError("模型未加载")

        model = self.model_manager.model
        vectorizer = self.model_manager.vectorizer
        mlb = self.model_manager.mlb

        X_vec = vectorizer.transform([text])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)
            labels = []
            for idx, score in enumerate(proba[0]):
                if score >= threshold:
                    name = mlb.classes_[idx]
                    labels.append(
                        {
                            "label": name,
                            "score": round(float(score), 4),
                            "solution": SOLUTION_KB.get(name, FALLBACK_SOLUTION),
                        }
                    )
            labels.sort(key=lambda x: x["score"], reverse=True)
            return labels[:max_labels]

        y_pred = model.predict(X_vec)
        return [
            {
                "label": str(name),
                "score": 1.0,
                "solution": SOLUTION_KB.get(str(name), FALLBACK_SOLUTION),
            }
            for idx, name in enumerate(mlb.classes_)
            if y_pred[0][idx] == 1
        ][:max_labels]

    def batch_predict(
        self,
        texts: List[str],
        threshold: float = INFERENCE_CONFIG["threshold"],
        max_labels: int = INFERENCE_CONFIG["max_labels"],
    ) -> List[List[dict]]:
        results = []
        for text in texts:
            try:
                result = self.predict(text, threshold, max_labels)
            except Exception as e:
                logger.error(f"批量预测单条失败: {e}")
                result = []
            results.append(result)
        return results


inference_service = InferenceService(model_manager)
app = Flask(__name__)

API_VERSION = "v1.0.0"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/version")
def version():
    return jsonify({
        "version": API_VERSION,
        "service": "IoT 智能家居多标签故障问答系统",
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if model_manager.is_ready() else "error",
        "model_loaded": model_manager.is_ready(),
        "labels": model_manager.get_labels(),
        "error": model_manager.load_error,
        "version": API_VERSION,
        "threshold": INFERENCE_CONFIG["threshold"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}

        if not isinstance(data, dict):
            return jsonify({
                "code": 400,
                "msg": "请求体必须为 JSON 对象",
                "data": [],
            }), 400

        text = data.get("text", "")
        if not text:
            return jsonify({
                "code": 400,
                "msg": "请提供 text 参数",
                "data": [],
            }), 400

        threshold = data.get("threshold", INFERENCE_CONFIG["threshold"])
        try:
            threshold = float(threshold)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("阈值必须在 0.0-1.0 之间")
        except ValueError as e:
            return jsonify({
                "code": 400,
                "msg": f"阈值参数无效: {e}",
                "data": [],
            }), 400

        validation = InputValidator.validate(text)
        if not validation["valid"]:
            return jsonify({
                "code": 400,
                "msg": validation["message"],
                "data": [],
            }), 400

        text = InputValidator.sanitize(text)

        if not model_manager.is_ready():
            return jsonify({
                "code": 503,
                "msg": "模型未加载，请检查模型文件",
                "data": [],
            }), 503

        logger.debug(f"推理请求: text='{text[:50]}...', threshold={threshold}")
        results = inference_service.predict(text, threshold)

        if not results:
            results = [{
                "label": "未识别",
                "score": 0.0,
                "solution": FALLBACK_SOLUTION,
            }]

        logger.info(f"推理完成: text='{text[:30]}...', labels={[r['label'] for r in results]}")

        return jsonify({
            "code": 0,
            "msg": "ok",
            "data": results,
            "threshold": threshold,
        })

    except Exception as e:
        logger.error(f"推理异常: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"服务器内部错误: {str(e)}",
            "data": [],
        }), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json(silent=True) or {}

        if not isinstance(data, dict):
            return jsonify({
                "code": 400,
                "msg": "请求体必须为 JSON 对象",
                "data": [],
            }), 400

        texts = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "code": 400,
                "msg": "请提供 texts 数组",
                "data": [],
            }), 400

        if len(texts) > INFERENCE_CONFIG["batch_size"]:
            return jsonify({
                "code": 400,
                "msg": f"单次批量请求最多 {INFERENCE_CONFIG['batch_size']} 条",
                "data": [],
            }), 400

        threshold = data.get("threshold", INFERENCE_CONFIG["threshold"])
        try:
            threshold = float(threshold)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("阈值必须在 0.0-1.0 之间")
        except ValueError as e:
            return jsonify({
                "code": 400,
                "msg": f"阈值参数无效: {e}",
                "data": [],
            }), 400

        if not model_manager.is_ready():
            return jsonify({
                "code": 503,
                "msg": "模型未加载，请检查模型文件",
                "data": [],
            }), 503

        sanitized_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            validation = InputValidator.validate(text)
            if validation["valid"]:
                sanitized_texts.append(InputValidator.sanitize(text))
                valid_indices.append(idx)
            else:
                logger.warning(f"批量请求第 {idx} 条校验失败: {validation['message']}")

        logger.debug(f"批量推理请求: {len(sanitized_texts)} 条")
        results = inference_service.batch_predict(sanitized_texts, threshold)

        final_results = []
        result_idx = 0
        for idx in range(len(texts)):
            if idx in valid_indices:
                final_results.append(results[result_idx])
                result_idx += 1
            else:
                final_results.append([{
                    "label": "校验失败",
                    "score": 0.0,
                    "solution": "输入文本不符合要求",
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

    except Exception as e:
        logger.error(f"批量推理异常: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"服务器内部错误: {str(e)}",
            "data": [],
        }), 500


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "code": 400,
        "msg": "请求参数错误",
        "data": [],
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "code": 404,
        "msg": "接口不存在",
        "data": [],
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "code": 500,
        "msg": "服务器内部错误",
        "data": [],
    }), 500


if __name__ == "__main__":
    app.run(
        host=FLASK_CONFIG["host"],
        port=FLASK_CONFIG["port"],
        debug=FLASK_CONFIG["debug"],
    )
