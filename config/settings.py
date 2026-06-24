"""
config/settings.py
==================
电商智能客服 NLU 意图识别引擎 — 全局配置管理

配置优先级：代码默认值 < .env 文件 < 环境变量

特性:
    1. 单一数据源(Single Source of Truth)：所有可调参数集中管理
    2. 环境变量覆盖：适合容器化部署与 CI/CD
    3. 类型安全：通过属性类型注解保证类型正确
    4. 路径自动创建：输出目录在首次访问时自动创建

业务定位:
    对标京东京小智、抖音飞鸽智能客服的 NLU 意图识别模块。
    上游: Project1 数据流水线提供清洗标注语料
    下游: 客服智能体根据意图做路由分发（RAG 问答 / 订单 API / 转人工）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 环境变量读取工具
# ============================================================

def _env_str(key: str, default: str) -> str:
    """读取字符串类型环境变量，不存在则返回默认值。"""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """读取整数类型环境变量，解析失败则返回默认值。"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"环境变量 {key}={val!r} 无法解析为 int，使用默认值 {default}")
        return default


def _env_float(key: str, default: float) -> float:
    """读取浮点数类型环境变量，解析失败则返回默认值。"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(f"环境变量 {key}={val!r} 无法解析为 float，使用默认值 {default}")
        return default


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔类型环境变量（true/1/yes -> True，其余 -> False）。"""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


# ============================================================
# 分类器配置常量（结构复杂，不适宜环境变量）
# ============================================================

CLASSIFIER_REGISTRY: Dict[str, dict] = {
    "logistic_regression": {
        "name": "逻辑回归",
        "model": "LogisticRegression",
        "params": {
            "max_iter": 1000,
            "C": 1.0,
            "solver": "lbfgs",
        },
        "grid_params": {
            "estimator__C": [0.1, 1.0, 10.0, 100.0],
            "estimator__solver": ["lbfgs", "saga"],
            "estimator__max_iter": [500, 1000, 2000],
        },
    },
    "random_forest": {
        "name": "随机森林",
        "model": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
        },
        "grid_params": {
            "estimator__n_estimators": [50, 100, 200],
            "estimator__max_depth": [None, 10, 20, 30],
            "estimator__min_samples_split": [2, 5, 10],
        },
    },
    "svm": {
        "name": "支持向量机",
        "model": "SVC",
        "params": {
            "kernel": "linear",
            "probability": True,
            "random_state": 42,
        },
        "grid_params": {
            "estimator__C": [0.1, 1.0, 10.0],
            "estimator__kernel": ["linear", "rbf"],
            "estimator__gamma": ["scale", "auto"],
        },
    },
}

DEFAULT_CLASSIFIER = "logistic_regression"

# ============================================================
# 意图路由策略（电商客服 NLU 核心：意图 → 下游路由）
# ============================================================
# 6 类电商咨询意图，对标京小智/飞鸽智能客服 NLU 模块
# 每个意图对应一条路由指令，上层客服系统据此做分发：
#   - RAG 问答链（售前商品咨询）
#   - 订单/物流 API 调用
#   - 工单系统
#   - 转人工坐席

INTENT_ROUTING: Dict[str, dict] = {
    "售前商品咨询": {
        "route": "rag_qa",
        "target": "商品知识库 RAG 问答链",
        "priority": "normal",
        "description": "用户询问商品属性、价格、优惠、库存、使用说明等售前问题，路由到商品知识库做检索增强生成",
        "example": "这款冰箱多少升？有什么优惠活动吗？",
    },
    "订单查询": {
        "route": "order_api",
        "target": "订单系统 API",
        "priority": "normal",
        "description": "用户查询订单状态、下单进度、订单详情、修改订单信息等，路由到订单中心拉取实时数据",
        "example": "我的订单什么时候发货？帮我查一下订单号 20240624001",
    },
    "物流查询": {
        "route": "logistics_api",
        "target": "物流接口 API",
        "priority": "normal",
        "description": "用户查询物流轨迹、配送进度、预计送达时间、快递公司信息等，路由到物流平台获取最新状态",
        "example": "快递到哪了？显示派送中三天了还没送到",
    },
    "售后退换货": {
        "route": "ticket_system",
        "target": "售后工单系统",
        "priority": "high",
        "description": "用户发起退货、换货、退款、维修等售后请求，自动创建工单并分配给售后处理组",
        "example": "收到的商品有划痕，我要退货退款",
    },
    "投诉纠纷": {
        "route": "human_agent",
        "target": "人工客服坐席",
        "priority": "urgent",
        "description": "用户表达强烈不满、投诉、索赔、威胁差评等，直接转接高级人工客服介入处理",
        "example": "你们这是虚假宣传！我要投诉到12315！",
    },
    "闲聊无关内容": {
        "route": "fallback",
        "target": "兜底回复 / 转人工",
        "priority": "low",
        "description": "问候、寒暄、测试消息、不相关话题等非业务咨询，返回礼貌兜底文案或静默转人工",
        "example": "你好呀 / 今天天气不错 / 发个表情包看看",
    },
}

FALLBACK_RESPONSE: str = (
    "您好，我是电商智能客服助手。以下信息可以帮助我更好地为您服务："
    "1) 您的订单号或商品名称；"
    "2) 您遇到的问题类型（下单/支付/物流/退换货等）；"
    "3) 相关截图或描述。"
    "如需人工服务，可直接回复「转人工」。"
)


# ============================================================
# 全局配置单例
# ============================================================

@dataclass
class Settings:
    """项目全局配置。

    所有参数可通过同名环境变量覆盖。
    实例化后通过单例 ``settings`` 全局访问。

    Example:
        from config.settings import settings
        print(settings.DATA_DIR)
    """

    # --- 基础路径 ---
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        _env_str("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ).resolve())
    DATA_DIR: Path = field(default_factory=lambda: Path(
        _env_str("DATA_DIR", "./data")
    ).resolve())
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(
        _env_str("OUTPUT_DIR", "./output")
    ).resolve())
    MODEL_DIR: Path = field(default_factory=lambda: Path(
        _env_str("MODEL_DIR", "./output/models")
    ).resolve())
    LOG_DIR: Path = field(default_factory=lambda: Path(
        _env_str("LOG_DIR", "./output/logs")
    ).resolve())

    # --- 数据文件路径 ---
    RAW_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env_str("RAW_DATA_PATH", "./data/labeled_data.csv")
    ).resolve())
    CLEANED_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env_str("CLEANED_DATA_PATH", "./data/cleaned_data.csv")
    ).resolve())

    # --- 模型命名 ---
    MODEL_NAME_PREFIX: str = field(default_factory=lambda: _env_str("MODEL_NAME_PREFIX", "best_model"))

    # --- TF-IDF 特征配置 ---
    TFIDF_MAX_FEATURES: int = field(default_factory=lambda: _env_int("TFIDF_MAX_FEATURES", 5000))
    TFIDF_NGRAM_MIN: int = field(default_factory=lambda: _env_int("TFIDF_NGRAM_MIN", 1))
    TFIDF_NGRAM_MAX: int = field(default_factory=lambda: _env_int("TFIDF_NGRAM_MAX", 2))
    TFIDF_ANALYZER: str = field(default_factory=lambda: _env_str("TFIDF_ANALYZER", "char"))

    # --- 训练参数 ---
    TRAIN_TEST_SIZE: float = field(default_factory=lambda: _env_float("TRAIN_TEST_SIZE", 0.2))
    TRAIN_VAL_SIZE: float = field(default_factory=lambda: _env_float("TRAIN_VAL_SIZE", 0.1))
    TRAIN_RANDOM_STATE: int = field(default_factory=lambda: _env_int("TRAIN_RANDOM_STATE", 42))

    # --- 推理默认值 ---
    INFERENCE_THRESHOLD: float = field(default_factory=lambda: _env_float("INFERENCE_THRESHOLD", 0.3))
    INFERENCE_MAX_LABELS: int = field(default_factory=lambda: _env_int("INFERENCE_MAX_LABELS", 10))
    INFERENCE_BATCH_SIZE: int = field(default_factory=lambda: _env_int("INFERENCE_BATCH_SIZE", 32))

    # --- Flask 服务 ---
    FLASK_HOST: str = field(default_factory=lambda: _env_str("FLASK_HOST", "0.0.0.0"))
    FLASK_PORT: int = field(default_factory=lambda: _env_int("FLASK_PORT", 5000))
    FLASK_DEBUG: bool = field(default_factory=lambda: _env_bool("FLASK_DEBUG", False))

    # --- 日志 ---
    LOG_LEVEL: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = field(default_factory=lambda: _env_str(
        "LOG_FORMAT",
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    ))
    LOG_DATE_FORMAT: str = field(default_factory=lambda: _env_str("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"))

    # ============================================================
    # 便捷属性
    # ============================================================

    @property
    def tfidf_config(self) -> Dict[str, object]:
        """返回 sklearn TfidfVectorizer 初始化参数字典。"""
        return {
            "max_features": self.TFIDF_MAX_FEATURES,
            "ngram_range": (self.TFIDF_NGRAM_MIN, self.TFIDF_NGRAM_MAX),
            "analyzer": self.TFIDF_ANALYZER,
        }

    @property
    def model_path(self) -> Path:
        """最佳模型文件路径。"""
        return self.MODEL_DIR / f"{self.MODEL_NAME_PREFIX}_model.pkl"

    @property
    def vectorizer_path(self) -> Path:
        """向量化器文件路径。"""
        return self.MODEL_DIR / f"{self.MODEL_NAME_PREFIX}_vectorizer.pkl"

    @property
    def mlb_path(self) -> Path:
        """标签二值化器文件路径。"""
        return self.MODEL_DIR / f"{self.MODEL_NAME_PREFIX}_mlb.pkl"

    # ============================================================
    # 目录自动创建
    # ============================================================

    def ensure_directories(self) -> None:
        """确保所有输出目录存在，通常在应用启动时调用。"""
        for dir_path in (self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR, self.OUTPUT_DIR):
            dir_path.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 表示
    # ============================================================

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  DATA_DIR={self.DATA_DIR},\n"
            f"  MODEL_DIR={self.MODEL_DIR},\n"
            f"  LOG_DIR={self.LOG_DIR},\n"
            f"  TRAIN_TEST_SIZE={self.TRAIN_TEST_SIZE},\n"
            f"  TRAIN_VAL_SIZE={self.TRAIN_VAL_SIZE},\n"
            f"  INFERENCE_THRESHOLD={self.INFERENCE_THRESHOLD},\n"
            f"  FLASK_HOST={self.FLASK_HOST},\n"
            f"  FLASK_PORT={self.FLASK_PORT},\n"
            f"  LOG_LEVEL={self.LOG_LEVEL},\n"
            f")"
        )


# ============================================================
# 全局单例
# ============================================================
settings = Settings()
