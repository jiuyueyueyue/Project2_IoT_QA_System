"""
config.py
=========
Project2_IoT_QA_System - 统一配置管理模块

包含：
- 路径配置（数据目录、模型目录、日志目录）
- 模型训练参数配置
- 推理服务配置
- 文本预处理配置
"""
import os

# ---------- 基础路径配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 数据文件路径
RAW_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

# 模型文件路径
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "best_model_vectorizer.pkl")
MLB_PATH = os.path.join(MODEL_DIR, "best_model_mlb.pkl")

# ---------- 文本预处理配置 ----------
TEXT_CONFIG = {
    "max_length": 500,           # 文本最大长度
    "min_length": 2,             # 文本最小长度
    "remove_punctuation": True,  # 是否去除标点符号
    "remove_stopwords": True,    # 是否去除停用词
    "lowercase": False,          # 是否转小写（中文不需要）
    "remove_extra_spaces": True, # 是否去除多余空格
}

# ---------- TF-IDF 配置 ----------
TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "analyzer": "char",
}

# ---------- 模型训练配置 ----------
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "validation_split": 0.1,
}

# ---------- 分类器配置 ----------
CLASSIFIERS = {
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

# 默认使用的分类器
DEFAULT_CLASSIFIER = "logistic_regression"

# ---------- 推理服务配置 ----------
INFERENCE_CONFIG = {
    "threshold": 0.3,            # 置信度阈值
    "max_labels": 10,            # 最大返回标签数
    "batch_size": 32,            # 批量处理大小
}

# ---------- Flask 服务配置 ----------
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
}

# ---------- 日志配置 ----------
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

# ---------- 故障方案知识库 ----------
SOLUTION_KB = {
    "空调": "检查遥控器电池与红外接收口；清洗滤网；确认设定温度与运行模式；如仍异常联系售后。",
    "客厅灯光": "检查智能开关联网状态；确认灯具是否断电；尝试 App 重新配对或重置场景。",
    "电动窗帘": "确认导轨供电与遥控器电量；检查电机是否有异响；必要时断电重启或手动复位。",
    "监控摄像头": "检查电源与 Wi-Fi 信号强度；确认 SD 卡状态；在 App 中重启设备并升级固件。",
    "扫地机器人": "清理滚刷与边刷、检查尘盒；确认充电桩通电；查看 App 报错码并按说明处理。",
    "参数调节": "可在 App 「设备设置」中调整对应参数；若调节无效，尝试恢复出厂设置后重新配置。",
    "故障报错": "查看设备 App 报错码并对照说明书；断电重启 30 秒后再次尝试；多次失败联系售后。",
    "联网配置": "确认路由器 2.4GHz 频段开启；将设备靠近路由器配网；按 App 提示完成 Wi-Fi 绑定。",
    "离线异常": "设备可能短暂掉线，等待 1-2 分钟自动恢复；如持续离线，检查路由器与设备电源。",
    "普通咨询": "可在 App 帮助中心查看产品使用指南，或联系在线客服获取详细说明。",
    "紧急故障": "请立即断电并停止使用，避免安全隐患；联系品牌官方售后安排上门维修。",
}

# 兜底文案
FALLBACK_SOLUTION = (
    "未识别到明确的故障类型，建议在描述中补充以下信息："
    "1) 故障设备名称；2) 故障现象与发生时间；3) 是否伴随异常声音/指示灯。"
)
