"""
train.py
=======
Project2_IoT_QA_System - 多标签故障文本分类模型训练脚本

功能特性：
1. 完整文本预处理（去符号/去空格/停用词）
2. 数据集去重清洗
3. 三类分类器（随机森林/SVM/逻辑回归）多模型对比实验
4. GridSearchCV 超参数调优
5. 三段划分（训练集/验证集/测试集）

命令行参数：
    --model          指定训练的模型类型: lr/rf/svm/all (默认: all)
    --grid-search    是否启用超参数调优 (默认: False)
    --compare        是否进行多模型对比 (默认: False)

示例：
    # 训练所有模型并对比
    python train.py --model all --compare

    # 使用超参数调优训练逻辑回归
    python train.py --model lr --grid-search

输出文件：
    models/best_model_model.pkl      : 最佳分类器模型
    models/best_model_vectorizer.pkl : TF-IDF 向量化器
    models/best_model_mlb.pkl        : MultiLabelBinarizer 标签转换器
"""
import argparse
import logging
import os

import pandas as pd

from config import (
    CLASSIFIERS,
    CLEANED_DATA_PATH,
    INFERENCE_CONFIG,
    LOG_CONFIG,
    RAW_DATA_PATH,
    TRAIN_CONFIG,
)
from utils.preprocess import (
    batch_preprocess,
    deduplicate_data,
    filter_valid_data,
    parse_sentiment,
)
from utils.train_eval import (
    compare_models,
    evaluate_model,
    fit_binarizer,
    grid_search_tune,
    save_model_artifacts,
    split_data,
    train_model,
    train_vectorizer,
)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"],
    datefmt=LOG_CONFIG["datefmt"],
)
logger = logging.getLogger("iot-qa-trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="IoT 智能家居多标签故障分类模型训练")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lr", "rf", "svm", "all"],
        help="训练的模型类型: lr(逻辑回归)/rf(随机森林)/svm(支持向量机)/all(全部)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="是否启用 GridSearchCV 超参数调优",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="是否进行多模型对比实验",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="交叉验证折数 (默认: 3)",
    )
    return parser.parse_args()


def load_and_preprocess_data():
    """加载并预处理数据"""
    logger.info(f"[1/6] 读取数据: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"      原始样本数: {len(df)}")

    df["labels"] = df["sentiment"].apply(parse_sentiment)
    df["text_clean"] = df["text"].apply(lambda x: batch_preprocess([x], remove_sw=False)[0])

    df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
    df = df[df["text_clean"].apply(len) > 0].reset_index(drop=True)
    logger.info(f"[2/6] 过滤空标签和短文本后样本数: {len(df)}")

    texts = df["text_clean"].tolist()
    labels = df["labels"].tolist()

    texts, labels = deduplicate_data(texts, labels)
    logger.info(f"[3/6] 去重后样本数: {len(texts)}")

    texts, labels = filter_valid_data(texts, labels)
    logger.info(f"[4/6] 过滤无效数据后样本数: {len(texts)}")

    df_clean = pd.DataFrame({"text": texts, "labels": labels})
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"      清洗后数据已保存: {CLEANED_DATA_PATH}")

    return texts, labels


def main():
    args = parse_args()

    texts, labels = load_and_preprocess_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        texts, labels,
        test_size=TRAIN_CONFIG["test_size"],
        val_size=TRAIN_CONFIG["validation_split"],
        random_state=TRAIN_CONFIG["random_state"],
    )

    logger.info(f"[5/6] 训练 TF-IDF 向量化器 ...")
    vectorizer = train_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    logger.info("[6/6] 拟合多标签二值化器 ...")
    mlb = fit_binarizer(y_train)
    y_train_mat = mlb.transform(y_train)
    y_val_mat = mlb.transform(y_val)
    y_test_mat = mlb.transform(y_test)

    if args.compare or args.model == "all":
        logger.info("\n" + "="*70)
        logger.info("多模型对比实验")
        logger.info("="*70)
        results_df = compare_models(
            X_train_vec, y_train_mat, X_test_vec, y_test_mat,
            use_grid_search=args.grid_search,
        )

        best_type = results_df.loc[0, "type"]
        logger.info(f"\n最佳模型: {results_df.loc[0, 'model']} (f1_micro={results_df.loc[0, 'f1_micro']:.4f})")
    else:
        model_map = {"lr": "logistic_regression", "rf": "random_forest", "svm": "svm"}
        best_type = model_map.get(args.model, "logistic_regression")
        logger.info(f"\n训练指定模型: {best_type}")

    logger.info("\n" + "="*70)
    logger.info(f"训练最终模型: {best_type}")
    logger.info("="*70)

    if args.grid_search:
        best_model, tune_results = grid_search_tune(
            X_train_vec, y_train_mat, best_type, cv=args.cv
        )
    else:
        best_model = train_model(X_train_vec, y_train_mat, best_type)
        tune_results = None

    logger.info("\n" + "="*70)
    logger.info("最终模型评估（测试集）")
    logger.info("="*70)
    final_metrics = evaluate_model(best_model, X_test_vec, y_test_mat)

    print("\n" + "="*70)
    print("训练总结")
    print("="*70)
    print(f"最佳模型: {CLASSIFIERS[best_type]['name']}")
    if tune_results:
        print(f"最佳参数: {tune_results['best_params']}")
        print(f"交叉验证 F1-micro: {tune_results['best_score']:.4f}")
    print(f"测试集评估:")
    print(f"  F1-micro:    {final_metrics['f1_micro']:.4f}")
    print(f"  F1-macro:    {final_metrics['f1_macro']:.4f}")
    print(f"  HammingLoss: {final_metrics['hamming_loss']:.4f}")
    print(f"  Jaccard:     {final_metrics['jaccard_micro']:.4f}")
    print("="*70)

    save_model_artifacts(best_model, vectorizer, mlb)

    logger.info("\n训练完成！")


if __name__ == "__main__":
    main()
