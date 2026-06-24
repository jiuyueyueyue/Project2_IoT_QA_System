"""
src/data_pipeline/loader.py
============================
数据加载器 — 将原始 CSV 数据加载为 pandas DataFrame。

变更说明（vs 旧 train.py）:
    - 旧: pd.read_csv(RAW_DATA_PATH) 散落在 train.py 主函数中
    - 新: 封装为 DataLoader 类，支持路径注入、格式校验、错误处理
    - 原因: 单一职责，消除全局变量依赖，便于测试

依赖:
    - pandas
    - src.utils.exceptions.DataLoadError
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class DataLoader:
    """原始数据加载器 — 从 CSV 文件加载标注数据集。

    Attributes:
        path: CSV 文件路径
        df: 加载后的 DataFrame（None 表示未加载）

    Example:
        loader = DataLoader("./data/labeled_data.csv")
        df = loader.load()
    """

    # 必需列名（缺失任一列则加载失败）
    REQUIRED_COLUMNS = {"text", "sentiment"}

    def __init__(self, path: Path | str) -> None:
        """初始化加载器。

        Args:
            path: CSV 数据文件路径
        """
        self.path = Path(path)
        self.df: Optional[pd.DataFrame] = None

    def load(self, encoding: str = "utf-8") -> pd.DataFrame:
        """加载 CSV 数据文件。

        Args:
            encoding: 文件编码，默认 utf-8

        Returns:
            加载的 DataFrame

        Raises:
            DataLoadError: 文件不存在 / 格式错误 / 缺少必需列
        """
        if not self.path.exists():
            raise DataLoadError(str(self.path), "文件不存在")

        logger.info(f"正在加载数据: {self.path}")

        try:
            self.df = pd.read_csv(self.path, encoding=encoding)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 解码失败，尝试 GBK 编码: {self.path}")
            try:
                self.df = pd.read_csv(self.path, encoding="gbk")
            except Exception as e:
                raise DataLoadError(str(self.path), f"编码错误: {e}") from e
        except Exception as e:
            raise DataLoadError(str(self.path), str(e)) from e

        # 校验必需列
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise DataLoadError(
                str(self.path),
                f"缺少必需列: {missing}，实际列: {list(self.df.columns)}",
            )

        logger.info(f"数据加载完成，共 {len(self.df)} 行，{len(self.df.columns)} 列")
        return self.df

    @property
    def n_rows(self) -> int:
        """已加载数据的行数。"""
        return len(self.df) if self.df is not None else 0
