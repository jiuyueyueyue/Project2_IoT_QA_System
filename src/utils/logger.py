"""
src/utils/logger.py
===================
统一日志配置 — 控制台 + 文件双输出，按日轮转。

使用方式:
    from src.utils.logger import setup_logger
    logger = setup_logger("my_module")
    logger.info("Hello")

特性:
    1. 控制台 handler：适合开发调试
    2. 文件 handler：输出到 output/logs/，按日轮转
    3. 等级由 Settings.LOG_LEVEL 控制
    4. 格式统一，包含时间戳、模块名、行号
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """创建并配置带有控制台和文件 handler 的 logger。

    Args:
        name: logger 名称（通常用 __name__）
        log_level: 日志等级 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_format: 日志格式字符串
        date_format: 日期格式
        log_dir: 日志文件目录，默认 output/logs/
        console: 是否启用控制台输出
        file_output: 是否启用文件输出

    Returns:
        配置完成的 logging.Logger 实例

    Example:
        from src.utils.logger import setup_logger
        logger = setup_logger(__name__)
        logger.info("模块初始化完成")
    """
    # 默认参数
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    if log_dir is None:
        from config.settings import settings
        log_dir = settings.LOG_DIR

    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)

    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 控制台 handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 handler（按日轮转，保留 7 天）
    if file_output:
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG 及以上
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到 root logger
    logger.propagate = False

    return logger
