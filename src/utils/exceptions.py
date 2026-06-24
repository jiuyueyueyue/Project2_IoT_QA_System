"""
src/utils/exceptions.py
=======================
项目自定义异常体系 — 便于精准捕获与日志记录。

设计思路:
    - 基类 ``ProjectException`` 可用于 catch-all 兜底
    - 子类语义化: DataLoadError / ModelNotFoundError / InferenceError / TrainingError
    - 每个异常携带上下文信息，便于排查

Example:
    try:
        load_data()
    except DataLoadError as e:
        logger.error(f"数据加载失败: {e}")
"""

from __future__ import annotations


class ProjectException(Exception):
    """项目基础异常，所有自定义异常继承自此。"""
    pass


class DataLoadError(ProjectException):
    """数据加载异常 — 文件不存在、格式错误、解析失败等。

    Args:
        path: 数据文件路径
        detail: 额外错误详情
    """

    def __init__(self, path: str, detail: str = "") -> None:
        self.path = path
        self.detail = detail
        msg = f"数据加载失败: {path}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class DataCleanError(ProjectException):
    """数据清洗异常 — 去重/过滤/解析过程中出现错误。"""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"数据清洗失败: {detail}" if detail else "数据清洗失败")


class ModelNotFoundError(ProjectException):
    """模型文件缺失异常 — 推理时找不到 pkl 文件。

    Args:
        path: 缺失的文件路径
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"模型文件不存在: {path}")


class ModelLoadError(ProjectException):
    """模型加载异常 — 文件存在但反序列化/校验失败。

    Args:
        path: 模型文件路径
        detail: 错误详情
    """

    def __init__(self, path: str, detail: str = "") -> None:
        self.path = path
        msg = f"模型加载失败: {path}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class InferenceError(ProjectException):
    """推理异常 — 推理过程中出现的错误。

    Args:
        detail: 错误详情
        text: 引发错误的输入文本（可选，已脱敏）
    """

    def __init__(self, detail: str = "", text: str = "") -> None:
        self.detail = detail
        self.text = text[:50] + "..." if len(text) > 50 else text
        msg = f"推理失败: {detail}"
        if self.text:
            msg += f" (输入: {self.text})"
        super().__init__(msg)


class TrainingError(ProjectException):
    """训练异常 — 模型训练过程中出现的错误。"""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"训练失败: {detail}" if detail else "训练失败")


class ValidationError(ProjectException):
    """输入验证异常。"""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"输入校验失败: {detail}" if detail else "输入校验失败")
