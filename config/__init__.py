"""
config — 项目配置层
===============
统一管理路径、模型参数、服务参数，支持环境变量覆盖。

使用方式:
    from config.settings import settings
    print(settings.DATA_DIR)
"""

from config.settings import Settings, settings

__all__ = ["Settings", "settings"]
