"""
utils/preprocess.py
===================
文本预处理工具模块

包含：
1. 标点符号去除
2. 多余空格去除
3. 停用词过滤
4. 文本规范化
5. 数据集清洗
"""
import json
import re
import string
from typing import List, Optional


CHINESE_STOPWORDS = {
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


def clean_text(text: str, max_length: int = 500, min_length: int = 2) -> str:
    """
    完整文本清洗流程

    Args:
        text: 原始文本
        max_length: 最大长度
        min_length: 最小长度

    Returns:
        清洗后的文本，过短返回空字符串
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if len(text) < min_length:
        return ""
    if len(text) > max_length:
        text = text[:max_length]

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)

    punctuation_pattern = re.compile(
        r"[{}]+".format(re.escape(string.punctuation + "，。！？、；：「」【】（）《》“”·～…—"))
    )
    text = punctuation_pattern.sub("", text)

    text = re.sub(r"\s+", "", text)

    return text


def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    """
    从文本中去除停用词

    Args:
        text: 已清洗的文本
        stopwords: 自定义停用词集合

    Returns:
        去除停用词后的文本
    """
    if not text:
        return text

    sw = stopwords or CHINESE_STOPWORDS
    filtered = [char for char in text if char not in sw and char != ' ']
    return ''.join(filtered)


def preprocess(text: str, max_length: int = 500, min_length: int = 2, 
               remove_sw: bool = False) -> str:
    """
    完整预处理流程：清洗 -> 去停用词(可选)

    Args:
        text: 原始文本
        remove_sw: 是否去除停用词

    Returns:
        预处理后的文本
    """
    text = clean_text(text, max_length, min_length)
    if remove_sw:
        text = remove_stopwords(text)
    return text


def batch_preprocess(texts: List[str], max_length: int = 500, min_length: int = 2,
                     remove_sw: bool = False) -> List[str]:
    """批量预处理文本"""
    return [preprocess(t, max_length, min_length, remove_sw) for t in texts]


def parse_sentiment(sentiment: str) -> List[str]:
    """
    解析 CSV 中 sentiment 列的 JSON 字符串

    Args:
        sentiment: JSON 字符串

    Returns:
        标签列表，解析失败返回空列表
    """
    try:
        obj = json.loads(sentiment)
        if isinstance(obj, dict):
            return obj.get("choices", []) or []
    except (ValueError, TypeError):
        pass
    return []


def deduplicate_data(texts: List[str], labels: List[List[str]]) -> tuple:
    """
    去重数据

    Args:
        texts: 文本列表
        labels: 标签列表

    Returns:
        (去重后的文本列表, 去重后的标签列表)
    """
    seen = set()
    unique_texts = []
    unique_labels = []

    for text, label_list in zip(texts, labels):
        if text and text not in seen:
            seen.add(text)
            unique_texts.append(text)
            unique_labels.append(label_list)

    return unique_texts, unique_labels


def filter_valid_data(texts: List[str], labels: List[List[str]], min_length: int = 2) -> tuple:
    """
    过滤有效数据

    Args:
        texts: 文本列表
        labels: 标签列表
        min_length: 文本最小长度

    Returns:
        (过滤后的文本列表, 过滤后的标签列表)
    """
    valid_texts = []
    valid_labels = []

    for text, label_list in zip(texts, labels):
        if text and len(text) >= min_length and label_list and len(label_list) > 0:
            valid_texts.append(text)
            valid_labels.append(label_list)

    return valid_texts, valid_labels
