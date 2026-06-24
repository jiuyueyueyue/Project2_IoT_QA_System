"""
src/features/preprocessor.py
=============================
文本预处理器 — 基于字符级的清洗、规范化、停用词过滤。

变更说明（vs 旧 utils/preprocess.py）:
    - 旧: 模块级函数，停用词集合散落在全局
    - 新: 封装为 TextPreprocessor 类，支持配置注入
    - 原因: 面向对象封装，支持不同的预处理策略组合

依赖:
    - re (内置)
    - string (内置)
"""

from __future__ import annotations

import logging
import re
import string
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================
# 中文停用词集（常用高频功能词）
# ============================================================

_CN_STOPWORDS: Set[str] = {
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "一个",
    "没有", "我们", "你们", "他们", "它们", "这个", "那个", "这些", "那些",
    "什么", "怎么", "如何", "为什么", "因为", "所以", "但是", "然而", "虽然",
    "如果", "要是", "只要", "只有", "除非", "无论", "不管", "尽管", "即使",
    "假如", "倘若", "万一", "可以", "能", "会", "应该", "必须", "需要",
    "可能", "也许", "大概", "差不多", "几乎", "完全", "十分", "非常", "很",
    "更", "最", "太", "极", "特别", "尤其", "稍微", "略微", "渐渐", "慢慢",
    "突然", "忽然", "已经", "曾经", "正在", "将要", "刚刚", "才", "又", "再",
    "也", "还", "全", "所有", "任何", "每", "各", "某", "一些", "许多",
    "大量", "少量", "少许", "一点", "几分", "几", "数", "若干", "众多",
    "不少", "无数", "一切", "全部", "整体", "整个", "全体", "各自", "每个",
    "每次", "各项", "各种", "各类", "各个", "各位", "各地", "各国", "等",
    "等等", "之类", "例如", "比如", "诸如", "像", "如", "好像", "仿佛",
    "似乎", "犹如", "宛如", "如同", "好比", "不如", "不及", "超过", "多于",
    "少于", "等于", "相当于", "约等于", "大约", "左右", "上下", "前后",
    "内外", "之间", "当中", "中间", "其中", "以内", "以上", "以下", "以外",
    "以前", "以后", "以来", "来", "去", "上", "下", "进", "出", "过", "起",
    "开", "关", "走", "跑", "跳", "飞", "游", "爬", "坐", "立", "躺", "卧",
    "站", "行", "说", "讲", "谈", "聊", "论", "议", "评", "述", "描", "写",
    "记", "录", "报", "告", "宣", "传", "公", "布", "发", "通", "知", "警",
    "告", "提", "醒", "指", "示", "命", "令", "要", "求", "请", "恳", "祈",
    "愿", "望", "希", "期", "盼", "企", "渴", "想", "需", "必", "应", "当",
    "须", "得", "肯", "可", "够", "不", "没", "无", "非", "否", "勿", "别",
    "莫", "休", "止", "停", "罢", "免", "禁", "阻", "防", "避", "躲", "逃",
    "藏", "隐", "掩", "遮", "蒙", "暗", "私", "窝", "收", "储", "保", "备",
    "预", "准", "安", "布", "摆", "陈", "展", "显", "表", "体", "反", "达",
    "传", "转", "翻", "解", "说", "阐", "分", "剖", "研", "探", "考", "追",
    "查", "调", "察", "监", "视", "控", "检", "验", "测", "试", "鉴", "评",
    "估", "价", "审", "计", "核", "批", "判", "裁", "决", "认", "确", "肯",
    "证", "指", "提", "辨", "识", "知", "懂", "明", "清", "透", "简", "复",
    "困", "容", "便", "快", "迅", "缓", "迟", "及", "准", "按", "定", "期",
    "偶", "时", "频", "少", "罕", "稀", "珍", "独", "奇", "新", "原", "自",
    "觉", "愿", "主", "被", "强", "勉", "甘", "乐", "答", "同", "支", "拥",
    "赞", "成", "反", "拒", "推", "脱", "溜", "消", "亡", "灭", "除", "清",
    "铲", "删", "修", "整", "调", "治", "统", "领", "导", "带", "引", "挥",
    "布", "顿", "置",
}


class TextPreprocessor:
    """文本预处理器 — 执行完整的文本清洗流水线。

    Attributes:
        max_length: 文本最大长度（超长截断）
        min_length: 文本最小长度（过短返回空）
        stopwords: 停用词集合

    Example:
        preprocessor = TextPreprocessor(max_length=500, min_length=2)
        clean = preprocessor.clean("空调不制冷，怎么办？")
    """

    def __init__(
        self,
        max_length: int = 500,
        min_length: int = 2,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """初始化预处理器。

        Args:
            max_length: 文本最大保留长度
            min_length: 有效文本最短长度
            stopwords: 自定义停用词集合，默认使用中文常用停用词
        """
        self.max_length = max_length
        self.min_length = min_length
        self.stopwords = stopwords if stopwords is not None else _CN_STOPWORDS

    # ============================================================
    # 公共方法
    # ============================================================

    def clean(self, text: str, remove_stopwords: bool = False) -> str:
        """对单条文本执行完整预处理流水线。

        流程: 空白处理 → 长度校验 → URL/HTML去除 → 标点去除 → 空格合并 → 停用词(可选)

        Args:
            text: 原始文本
            remove_stopwords: 是否去除停用词

        Returns:
            清洗后的文本（过短返回空字符串）
        """
        if not isinstance(text, str):
            return ""

        # 1. 去首尾空白
        text = text.strip()
        if not text:
            return ""

        # 2. 长度校验
        if len(text) < self.min_length:
            return ""
        if len(text) > self.max_length:
            text = text[:self.max_length]

        # 3. 去除 URL 和 HTML 标签
        text = self._remove_urls(text)
        text = self._remove_html_tags(text)

        # 4. 去除标点符号（中英文）
        text = self._remove_punctuation(text)

        # 5. 合并/去除多余空格
        text = self._normalize_spaces(text)

        # 6. 可选：去除停用词
        if remove_stopwords:
            text = self._remove_stopwords(text)

        return text

    def batch_clean(
        self, texts: List[str], remove_stopwords: bool = False
    ) -> List[str]:
        """批量文本预处理。

        Args:
            texts: 原始文本列表
            remove_stopwords: 是否去除停用词

        Returns:
            清洗后文本列表
        """
        return [self.clean(t, remove_stopwords=remove_stopwords) for t in texts]

    # ============================================================
    # 私有清洗方法
    # ============================================================

    @staticmethod
    def _remove_urls(text: str) -> str:
        """去除 HTTP/HTTPS URL。"""
        return re.sub(r"http\S+", "", text)

    @staticmethod
    def _remove_html_tags(text: str) -> str:
        """去除 HTML/XML 标签。"""
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """去除中英文标点符号。"""
        punctuation_chars = string.punctuation + "，。！？、；：「」【】（）《》"·～…—"
        pattern = re.compile(rf"[{re.escape(punctuation_chars)}]+")
        return pattern.sub("", text)

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        """压缩连续空格为单个空格，再去除所有空格（中文场景）。"""
        return re.sub(r"\s+", "", text)

    def _remove_stopwords(self, text: str) -> str:
        """从已清洗文本中去除停用词。"""
        if not text:
            return text
        return "".join(char for char in text if char not in self.stopwords)

    # ============================================================
    # 轻量清洗（推理时使用，不去标点，仅规范化）
    # ============================================================

    def sanitize(self, text: str) -> str:
        """推理前轻量清洗 — 仅去首尾空、合并空格、截断。

        与 clean() 的区别: 不去标点、不去 URL，保留语义完整性。

        Args:
            text: 用户输入原始文本

        Returns:
            规范化后的文本
        """
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > self.max_length:
            text = text[:self.max_length]
        return text
