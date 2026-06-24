# 电商智能客服 NLU 意图识别引擎

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

> 🎯 **对标产品**: 京东京小智 NLU 模块 / 抖音飞鸽智能客服意图理解层

多标签文本意图分类引擎，输入用户消息 → 输出意图类别 + 置信度 + 路由分发策略，供上层客服系统做智能路由。

## 📍 项目定位

本系统是**电商智能客服架构中的 NLU（自然语言理解）核心模块**，承担"理解用户意图并分发到对应处理单元"的职责。

```
┌─────────────────────────────────────────────────────┐
│                  电商智能客服整体架构                    │
├───────────┬──────────────┬──────────────┬────────────┤
│ Project 1 │  Project 2   │  Project 3   │  对外服务   │
│ 数据流水线 │  NLU 意图引擎 │ 客服智能体    │            │
│           │   (本项目)    │              │            │
├───────────┼──────────────┼──────────────┼────────────┤
│ 数据采集   │  意图分类     │  RAG 问答链   │  用户端     │
│ 语料清洗   │  置信度计算   │  订单 API     │  Web/App   │
│ 标注管理   │  路由决策     │  转人工逻辑   │            │
└───────────┴──────────────┴──────────────┴────────────┘
     上游           本项目          下游
```

| 方向 | 说明 |
|------|------|
| 上游 | Project1 数据流水线 → 提供清洗后的标注语料（CSV 格式） |
| 本模块 | 训练意图分类模型 → 暴露 API 供调用 |
| 下游 | 客服智能体根据 `route` 字段做分发：`rag_qa` / `order_api` / `logistics_api` / `ticket_system` / `human_agent` |

## 🏗️ 意图类别与路由策略

系统识别 **6 类电商咨询意图**，每类对应一个下游路由：

| # | 意图类别 | 路由 Key | 路由目标 | 优先级 | 示例 |
|---|----------|----------|----------|--------|------|
| 1 | 售前商品咨询 | `rag_qa` | 商品知识库 RAG 问答 | normal | "这款手机有什么颜色" |
| 2 | 订单查询 | `order_api` | 订单系统 API | normal | "我的订单什么时候发货" |
| 3 | 物流查询 | `logistics_api` | 物流接口 API | normal | "快递到哪了怎么还没到" |
| 4 | 售后退换货 | `ticket_system` | 售后工单系统 | high | "收到有划痕我要退货退款" |
| 5 | 投诉纠纷 | `human_agent` | 人工客服坐席 | urgent | "你们这是虚假宣传我要投诉" |
| 6 | 闲聊无关内容 | `fallback` | 兜底回复/转人工 | low | "今天天气不错哈哈" |

## 📁 项目结构

```
Project2_IoT_QA_System/
├── main.py                    # 统一 CLI 入口 (train/serve/predict/info)
├── app.py                     # Flask API 路由层
├── config/
│   ├── settings.py            # 全局配置 (Settings 类 + 环境变量)
│   └── .env.example           # 环境变量模板
├── src/
│   ├── data_pipeline/         # 数据加载 + 清洗
│   ├── features/              # 文本预处理 + TF-IDF 向量化
│   ├── training/              # 模型训练 + 多标签评估
│   ├── inference/             # 模型管理 + 意图推理
│   └── utils/                 # 日志 + 自定义异常
├── data/
│   └── labeled_data.csv       # 120 条电商意图标注数据
├── output/
│   ├── models/                # 训练产出的模型文件
│   └── logs/                  # 运行日志
├── templates/index.html       # 意图识别测试页面
├── static/css/style.css       # 前端样式
└── README.md
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练意图分类模型
python main.py train

# 启动推理服务
python main.py serve

# CLI 快速测试
python main.py predict --text "我的订单什么时候发货"

# 查看系统信息
python main.py info
```

启动后访问 `http://127.0.0.1:5000/` 使用 Web 测试界面。

## 📡 API 接口

### 意图识别（单条）

```bash
POST /api/predict
Content-Type: application/json

{"text": "我要退货退款", "threshold": 0.3}
```

```json
{
  "code": 0,
  "msg": "ok",
  "data": [
    {
      "label": "售后退换货",
      "score": 0.8512,
      "action": {
        "route": "ticket_system",
        "target": "售后工单系统",
        "priority": "high",
        "description": "用户发起退货、换货、退款、维修等售后请求..."
      }
    }
  ],
  "threshold": 0.3
}
```

### 批量识别

```bash
POST /api/batch
Content-Type: application/json

{"texts": ["快递在哪", "我要退货"], "threshold": 0.3}
```

### 健康检查

```bash
GET /api/health
```

返回模型状态、支持的全部意图类别、路由映射表。

## 📊 评估指标

| 指标 | 说明 |
|------|------|
| Subset Accuracy | 精确匹配率（所有标签全对的比例） |
| Hamming Loss | 汉明损失（越小越好） |
| F1-micro/macro/weighted | 多标签 F1 分数 |
| Precision/Recall-micro | 微平均精确率/召回率 |
| Jaccard-micro/macro | 多标签 Jaccard 相似度 |
| Per-Class P/R/F1 | 每个意图类别的独立评估 |

## 🔧 配置说明

所有参数通过 `config/settings.py` 集中管理，支持 `.env` 环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TFIDF_MAX_FEATURES` | 5000 | TF-IDF 特征维度 |
| `TRAIN_TEST_SIZE` | 0.2 | 测试集占比 |
| `INFERENCE_THRESHOLD` | 0.3 | 默认置信度阈值 |
| `FLASK_HOST` | 0.0.0.0 | 服务绑定地址 |
| `FLASK_PORT` | 5000 | 服务端口 |
| `LOG_LEVEL` | INFO | 日志等级 |

## 🛣️ 模块扩展思路

1. **模型升级**: TF-IDF + 传统 ML → BERT/RoBERTa 微调，提升语义理解
2. **数据增强**: 120 条 → 1000+ 条，覆盖更多口语化表达
3. **多轮对话**: 结合对话历史做意图消歧与上下文关联
4. **在线学习**: 用户纠错反馈 → 模型在线迭代
5. **Docker 部署**: 容器化 + CI/CD，支持一键上线
6. **A/B 实验**: 多模型灰度对比，数据驱动选型

## 📜 许可证

MIT License
