# IoT 智能家居多标签故障问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

一个基于 Flask 的 IoT 智能家居多标签故障问答系统，支持自动识别故障设备类型并提供维修解决方案。

## 项目背景

随着智能家居设备的普及，用户遇到故障时往往不知道该联系哪个设备的售后，或者不知道如何自行排查。本系统通过自然语言处理技术，自动分析用户输入的故障描述，识别出涉及的设备类型和故障类型，并提供对应的维修建议。

## 技术栈

- **后端框架**: Flask 2.0+
- **机器学习**: scikit-learn 1.0+
- **数据处理**: pandas, numpy
- **前端**: HTML5, CSS3, JavaScript
- **模型持久化**: pickle

## 项目结构

```
Project2_IoT_QA_System/
├── app.py                    # Flask 后端服务入口
├── train.py                  # 模型训练脚本
├── config.py                 # 全局配置管理
├── requirements.txt          # 项目依赖清单
├── README.md                 # 项目说明文档
├── data/
│   └── labeled_data.csv      # 标注数据集（106条样本）
├── models/                   # 训练产出模型文件
│   ├── best_model_model.pkl
│   ├── best_model_vectorizer.pkl
│   └── best_model_mlb.pkl
├── utils/
│   ├── preprocess.py         # 文本预处理工具
│   └── train_eval.py         # 训练评估工具
├── templates/
│   └── index.html            # 前端页面
└── static/
    └── css/
        └── style.css         # 样式文件
```

## 数据集说明

### 数据来源

数据集为模拟智能家居故障咨询场景构建，包含 106 条标注样本。

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

### 数据格式

CSV 文件格式：
```csv
annotation_id,annotator,created_at,id,lead_time,sentiment,source,text,updated_at
```

其中 `sentiment` 字段为 JSON 格式，包含 `choices` 数组表示多标签：
```json
{"choices":["空调","故障报错"]}
```

## 模型训练

### 运行训练

```bash
# 训练逻辑回归模型（默认）
python train.py --model lr

# 训练所有模型并对比
python train.py --model all --compare

# 使用超参数调优
python train.py --model lr --grid-search
```

### 训练流程

1. **数据加载**: 读取 CSV 数据集
2. **文本预处理**: 去除标点、空格、停用词
3. **数据清洗**: 去重、过滤无效数据
4. **数据集划分**: 训练集(70%)/验证集(10%)/测试集(20%)
5. **特征提取**: TF-IDF 字符级向量化
6. **标签编码**: MultiLabelBinarizer 多标签编码
7. **模型训练**: OneVsRest + 基分类器
8. **模型评估**: F1-micro/macro、HammingLoss、Jaccard
9. **模型保存**: 序列化到 models/ 目录

### 多模型对比实验

| 模型 | F1-micro | F1-macro | HammingLoss | Jaccard |
|------|----------|----------|-------------|---------|
| 逻辑回归 | 0.1778 | 0.1394 | 0.1529 | 0.0976 |
| 随机森林 | - | - | - | - |
| SVM | - | - | - | - |

### 超参数调优

使用 GridSearchCV 进行网格搜索：

**逻辑回归搜索空间**:
- `estimator__C`: [0.1, 1.0, 10.0, 100.0]
- `estimator__solver`: ['lbfgs', 'saga']
- `estimator__max_iter`: [500, 1000, 2000]

**最佳参数**: `{'estimator__C': 0.1, 'estimator__max_iter': 1000, 'estimator__solver': 'saga'}`

## API 接口

### 健康检查

```bash
GET /api/health
```

响应示例：
```json
{
    "status": "ok",
    "model_loaded": true,
    "labels": ["参数调节", "客厅灯光", "扫地机器人", ...],
    "threshold": 0.3,
    "version": "v1.0.0"
}
```

### 单条预测

```bash
POST /api/predict
Content-Type: application/json

{"text": "扫地机器人无法充电", "threshold": 0.3}
```

响应示例：
```json
{
    "code": 0,
    "msg": "ok",
    "data": [
        {
            "label": "扫地机器人",
            "score": 0.5926,
            "solution": "清理滚刷与边刷、检查尘盒；确认充电桩通电；查看 App 报错码并按说明处理。"
        }
    ],
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

## 前端页面

访问 `http://127.0.0.1:5000/` 即可使用问答界面：

- **输入框**: 输入故障描述文本
- **识别按钮**: 触发模型推理
- **结果展示**: 显示识别出的设备标签、置信度和维修建议

## 部署启动

### 环境要求

- Python 3.8+
- pip 20.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py --model lr
```

### 启动服务

```bash
python app.py
```

### 访问地址

- 前端页面: `http://127.0.0.1:5000/`
- API 接口: `http://127.0.0.1:5000/api/health`

## 优化迭代思路

### 已完成优化

1. **数据层**: 扩充数据集至 106 条，实现类别均衡
2. **预处理**: 完整文本预处理流水线（去符号、去空格、停用词过滤）
3. **模型对比**: 支持逻辑回归、随机森林、SVM 三类分类器对比
4. **超参数调优**: GridSearchCV 自动搜索最优参数组合
5. **工程化重构**: 模块化目录结构，消除硬编码
6. **后端健壮性**: 输入校验、异常捕获、分级日志、阈值校验
7. **前端适配**: 置信度滑块可调，优化展示效果

### 后续可拓展方向

1. **模型升级**: 引入 BERT/RoBERTa 等预训练语言模型，提升语义理解能力
2. **数据增强**: 使用数据增强技术扩充数据集（同义词替换、回译等）
3. **在线学习**: 支持用户反馈，实现模型在线迭代更新
4. **多轮对话**: 实现对话式故障诊断，逐步引导用户补充信息
5. **知识图谱**: 构建设备故障知识图谱，提供更精准的解决方案
6. **容器化部署**: Docker 容器化，支持一键部署
7. **性能优化**: 模型量化、推理加速，支持高并发场景

## 运行截图

### 首页界面

![首页](docs/screenshots/home.png)

### 识别结果

![识别结果](docs/screenshots/result.png)

## 许可证

MIT License

## 作者

用于个人求职作品集展示
