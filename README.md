# IoT 智能家居多标签故障问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

一个基于 Flask + scikit-learn 的 IoT 智能家居多标签故障问答系统，支持自动识别故障设备类型并提供维修解决方案。

> 🏗️ **v2.0 企业级工程重构** — 分层架构 · 配置分离 · 类型安全 · 评估完善

## 📋 项目背景

随着智能家居设备的普及，用户遇到故障时往往不知道该联系哪个设备的售后。本系统通过多标签文本分类技术，自动分析用户输入的故障描述，识别涉及的设备类型和故障类型，并提供对应维修建议。

## 🏛️ 架构设计

```
Project2_IoT_QA_System/
├── main.py                          # 统一 CLI 入口 (train/serve/predict/info)
├── app.py                           # Flask 路由层（薄控制器）
├── config/                          # 配置层
│   ├── settings.py                  # 全局配置类（环境变量覆盖）
│   └── .env.example                 # 环境变量模板
├── src/                             # 核心业务模块
│   ├── data_pipeline/               # 数据流水线
│   │   ├── loader.py                # 数据加载器
│   │   └── cleaner.py               # 数据清洗器
│   ├── features/                    # 特征工程
│   │   ├── preprocessor.py          # 文本预处理器
│   │   └── vectorizer.py            # TF-IDF 向量化管理器
│   ├── training/                    # 训练模块
│   │   ├── trainer.py               # 训练器类
│   │   └── evaluator.py             # 多标签评估器
│   ├── inference/                   # 推理模块
│   │   ├── model_manager.py         # 模型管理器
│   │   └── predictor.py             # 推理服务
│   └── utils/                       # 工具模块
│       ├── logger.py                # 统一日志配置
│       └── exceptions.py            # 自定义异常体系
├── data/                            # 数据目录
├── output/                          # 输出目录
│   ├── models/                      # 模型产物
│   └── logs/                        # 日志文件
├── templates/                       # 前端模板
├── static/                          # 静态资源
├── requirements.txt                 # 依赖清单
├── .gitignore                       # Git 忽略规则
└── README.md                        # 项目文档
```

### 分层职责

| 层级 | 目录 | 职责 |
|------|------|------|
| 入口层 | `main.py` / `app.py` | CLI 命令分发 / HTTP 请求响应 |
| 配置层 | `config/` | 统一参数管理，环境变量覆盖 |
| 数据流水线 | `src/data_pipeline/` | 数据加载、解析、清洗、去重 |
| 特征工程 | `src/features/` | 文本预处理、TF-IDF 向量化 |
| 训练 | `src/training/` | 模型训练、超参调优、多模型对比 |
| 推理 | `src/inference/` | 模型加载、预测推理 |
| 工具 | `src/utils/` | 日志、异常、通用工具 |
| 输出 | `output/` | 模型文件、日志持久化 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 20.0+

### 安装

```bash
# 克隆项目
git clone <repo-url> && cd Project2_IoT_QA_System

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 复制环境变量模板（可选）
cp config/.env.example .env
```

### 训练模型

```bash
# 训练逻辑回归（默认）
python main.py train

# 训练指定模型
python main.py train --model rf

# 多模型对比实验
python main.py train --all --compare

# 超参数调优
python main.py train --model lr --grid-search
```

### 启动服务

```bash
# 启动 Flask 推理服务
python main.py serve

# 自定义主机/端口
python main.py serve --host 127.0.0.1 --port 8080
```

访问 `http://127.0.0.1:5000/` 使用问答界面。

### CLI 推理

```bash
python main.py predict --text "空调不制冷怎么办"
```

### 查看系统信息

```bash
python main.py info
```

## 📊 数据集

### 数据来源

模拟智能家居故障咨询场景构建，共 106 条标注样本。

### 标签类别（11类）

| 类别 | 描述 | 样本数 |
|------|------|--------|
| 空调 | 空调设备相关 | 12 |
| 客厅灯光 | 智能灯光相关 | 12 |
| 电动窗帘 | 电动窗帘相关 | 12 |
| 监控摄像头 | 监控摄像头相关 | 12 |
| 扫地机器人 | 扫地机器人相关 | 12 |
| 参数调节 | 参数设置相关 | 8 |
| 故障报错 | 设备故障报错 | 11 |
| 联网配置 | WiFi配网相关 | 8 |
| 离线异常 | 设备离线问题 | 8 |
| 普通咨询 | 使用咨询 | 8 |
| 紧急故障 | 紧急故障情况 | 5 |

## 📡 API 接口

### 健康检查

```bash
GET /api/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "labels": ["空调", "客厅灯光", "扫地机器人", ...],
  "n_labels": 11,
  "threshold": 0.3,
  "version": "v2.0.0"
}
```

### 单条预测

```bash
POST /api/predict
Content-Type: application/json

{"text": "扫地机器人无法充电", "threshold": 0.3}
```

```json
{
  "code": 0,
  "msg": "ok",
  "data": [{
    "label": "扫地机器人",
    "score": 0.5926,
    "solution": "清理滚刷与边刷、检查尘盒；确认充电桩通电；查看 App 报错码并按说明处理。"
  }],
  "threshold": 0.3
}
```

### 批量预测

```bash
POST /api/batch
Content-Type: application/json

{"texts": ["空调不制冷", "灯光闪烁"], "threshold": 0.3}
```

### 版本信息

```bash
GET /api/version
```

## 📈 评估指标

多标签分类采用以下评估体系：

| 指标 | 说明 |
|------|------|
| Subset Accuracy | 精确匹配率（最严格：所有标签全对才算对） |
| Hamming Loss | 汉明损失（越小越好） |
| F1-micro | 全局微平均 F1 |
| F1-macro | 各类别宏平均 F1 |
| F1-weighted | 按样本数加权的 F1 |
| Jaccard-micro | 微平均 Jaccard |
| Per-Class P/R/F1 | 每个标签的精确率/召回率/F1 |

## 🔧 配置说明

所有配置通过 `config/settings.py` 管理，支持环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TFIDF_MAX_FEATURES` | 5000 | TF-IDF 最大特征数 |
| `TRAIN_TEST_SIZE` | 0.2 | 测试集占比 |
| `TRAIN_VAL_SIZE` | 0.1 | 验证集占比 |
| `INFERENCE_THRESHOLD` | 0.3 | 默认置信度阈值 |
| `FLASK_HOST` | 0.0.0.0 | Flask 绑定地址 |
| `FLASK_PORT` | 5000 | Flask 端口 |
| `LOG_LEVEL` | INFO | 日志等级 |

详见 `config/.env.example`。

## 🛣️ 后续拓展

- [ ] 引入 BERT/RoBERTa 预训练模型提升语义理解
- [ ] 数据增强扩充数据集（同义词替换、回译）
- [ ] 在线学习（用户反馈闭环）
- [ ] 多轮对话故障诊断
- [ ] 知识图谱增强
- [ ] Docker 容器化部署
- [ ] 模型量化与推理加速
- [ ] 单元测试与 CI/CD

## 📜 许可证

MIT License

## 👤 作者

用于个人求职作品集展示
