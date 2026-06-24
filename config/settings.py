"""
config/settings.py
==================
全局配置管理类 — 所有硬编码路径/参数集中于此。

配置优先级：代码默认值 < .env 文件 < 环境变量

特性:
    1. 单一数据源(Single Source of Truth)：所有可调参数集中管理
    2. 环境变量覆盖：适合容器化部署与 CI/CD
    3. 类型安全：通过属性类型注解保证类型正确
    4. 路径自动创建：输出目录在首次访问时自动创建
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
# 故障方案知识库（业务常量）
# ============================================================

SOLUTION_KB: Dict[str, str] = {
    "空调": "检查遥控器电池与红外接收口；清洗滤网；确认设定温度与运行模式；如仍异常联系售后。",
    "客厅灯光": "检查智能开关联网状态；确认灯具是否断电；尝试 App 重新配对或重置场景。",
    "电动窗帘": "确认导轨供电与遥控器电量；检查电机是否有异响；必要时断电重启或手动复位。",
    "监控摄像头": "检查电源与 Wi-Fi 信号强度；确认 SD 卡状态；在 App 中重启设备并升级固件。",
    "扫地机器人": "清理滚刷与边刷、检查尘盒；确认充电桩通电；查看 App 报错码并按说明处理。",
    "参数调节": "可在 App「设备设置」中调整对应参数；若调节无效，尝试恢复出厂设置后重新配置。",
    "故障报错": "查看设备 App 报错码并对照说明书；断电重启 30 秒后再次尝试；多次失败联系售后。",
    "联网配置": "确认路由器 2.4GHz 频段开启；将设备靠近路由器配网；按 App 提示完成 Wi-Fi 绑定。",
    "离线异常": "设备可能短暂掉线，等待 1-2 分钟自动恢复；如持续离线，检查路由器与设备电源。",
    "普通咨询": "可在 App 帮助中心查看产品使用指南，或联系在线客服获取详细说明。",
    "紧急故障": "请立即断电并停止使用，避免安全隐患；联系品牌官方售后安排上门维修。",
}

FALLBACK_SOLUTION: str = (
    "未识别到明确的故障类型，建议在描述中补充以下信息："
    "1) 故障设备名称；2) 故障现象与发生时间；3) 是否伴随异常声音/指示灯。"
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
