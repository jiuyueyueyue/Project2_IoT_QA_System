"""src/utils — 工具模块

提供:
    - setup_logger: 统一日志配置
    - 自定义异常类体系
"""

from src.utils.exceptions import (
    DataCleanError,
    DataLoadError,
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ProjectException,
    TrainingError,
    ValidationError,
)
from src.utils.logger import setup_logger

__all__ = [
    "setup_logger",
    "ProjectException",
    "DataLoadError",
    "DataCleanError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "TrainingError",
    "ValidationError",
]
