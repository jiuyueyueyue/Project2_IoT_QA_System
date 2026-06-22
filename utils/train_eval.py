"""
utils/train_eval.py
===================
模型训练与评估工具模块

包含：
1. 多标签分类器训练
2. GridSearchCV 超参数调优
3. 多模型对比实验
4. 模型评估指标计算
"""
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from config import CLASSIFIERS, MODEL_DIR, TFIDF_CONFIG

logger = logging.getLogger(__name__)


def train_vectorizer(texts: List[str], config: Optional[Dict] = None) -> TfidfVectorizer:
    """
    训练 TF-IDF 向量化器

    Args:
        texts: 文本列表
        config: TF-IDF 配置

    Returns:
        训练好的向量化器
    """
    cfg = config or TFIDF_CONFIG
    logger.info(f"训练 TF-IDF 向量化器，配置: {cfg}")

    vectorizer = TfidfVectorizer(**cfg)
    vectorizer.fit(texts)

    logger.info(f"向量化器训练完成，特征数: {len(vectorizer.get_feature_names_out())}")
    return vectorizer


def fit_binarizer(labels: List[List[str]]) -> MultiLabelBinarizer:
    """
    拟合多标签二值化器

    Args:
        labels: 标签列表

    Returns:
        拟合好的二值化器
    """
    logger.info("拟合 MultiLabelBinarizer")

    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    logger.info(f"标签类别数: {len(mlb.classes_)} -> {list(mlb.classes_)}")
    return mlb


def create_classifier(classifier_type: str) -> OneVsRestClassifier:
    """
    创建指定类型的分类器

    Args:
        classifier_type: 分类器类型

    Returns:
        OneVsRestClassifier 包装的分类器
    """
    config = CLASSIFIERS.get(classifier_type)
    if not config:
        raise ValueError(f"未知分类器类型: {classifier_type}")

    base_model = None
    model_name = config["model"]

    if model_name == "LogisticRegression":
        base_model = LogisticRegression(**config["params"])
    elif model_name == "RandomForestClassifier":
        base_model = RandomForestClassifier(**config["params"])
    elif model_name == "SVC":
        base_model = SVC(**config["params"])
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return OneVsRestClassifier(base_model)


def train_model(X_vec, y_mat, classifier_type: str = "logistic_regression") -> OneVsRestClassifier:
    """
    训练分类器

    Args:
        X_vec: 特征向量矩阵
        y_mat: 标签矩阵
        classifier_type: 分类器类型

    Returns:
        训练好的分类器
    """
    logger.info(f"训练分类器: {classifier_type} ({CLASSIFIERS[classifier_type]['name']})")

    model = create_classifier(classifier_type)
    model.fit(X_vec, y_mat)

    logger.info("分类器训练完成")
    return model


def grid_search_tune(
    X_vec,
    y_mat,
    classifier_type: str = "logistic_regression",
    cv: int = 3,
    n_jobs: int = -1,
) -> Tuple[OneVsRestClassifier, Dict]:
    """
    使用 GridSearchCV 进行超参数调优

    Args:
        X_vec: 特征向量矩阵
        y_mat: 标签矩阵
        classifier_type: 分类器类型
        cv: 交叉验证折数
        n_jobs: 并行工作数

    Returns:
        (最佳模型, 调优结果)
    """
    config = CLASSIFIERS.get(classifier_type)
    if not config:
        raise ValueError(f"未知分类器类型: {classifier_type}")

    logger.info(f"开始 GridSearchCV 超参数调优: {classifier_type}")
    logger.info(f"搜索参数: {config['grid_params']}")

    model = create_classifier(classifier_type)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=config["grid_params"],
        cv=cv,
        scoring="f1_micro",
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X_vec, y_mat)

    best_model = grid_search.best_estimator_

    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }

    logger.info(f"超参数调优完成，最佳参数: {results['best_params']}")
    logger.info(f"最佳交叉验证 F1-micro: {results['best_score']:.4f}")

    return best_model, results


def evaluate_model(model, X_test_vec, y_test_mat) -> Dict:
    """
    评估模型性能

    Args:
        model: 模型
        X_test_vec: 测试特征向量
        y_test_mat: 测试标签矩阵

    Returns:
        评估指标字典
    """
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test_mat, y_pred),
        "f1_micro": f1_score(y_test_mat, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_test_mat, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test_mat, y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(y_test_mat, y_pred),
        "jaccard_micro": jaccard_score(y_test_mat, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_test_mat, y_pred, average="macro", zero_division=0),
    }

    logger.info(f"评估结果: {metrics}")

    return metrics


def compare_models(
    X_train_vec,
    y_train_mat,
    X_test_vec,
    y_test_mat,
    classifier_types: Optional[List[str]] = None,
    use_grid_search: bool = False,
) -> pd.DataFrame:
    """
    多模型对比实验

    Args:
        X_train_vec: 训练特征向量
        y_train_mat: 训练标签矩阵
        X_test_vec: 测试特征向量
        y_test_mat: 测试标签矩阵
        classifier_types: 要对比的分类器类型列表
        use_grid_search: 是否使用超参数调优

    Returns:
        模型对比结果 DataFrame
    """
    types = classifier_types or list(CLASSIFIERS.keys())
    results = []

    for classifier_type in types:
        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型: {classifier_type} ({CLASSIFIERS[classifier_type]['name']})")
        logger.info(f"{'='*60}")

        if use_grid_search:
            model, _ = grid_search_tune(X_train_vec, y_train_mat, classifier_type)
        else:
            model = train_model(X_train_vec, y_train_mat, classifier_type)

        metrics = evaluate_model(model, X_test_vec, y_test_mat)

        results.append({
            "model": CLASSIFIERS[classifier_type]["name"],
            "type": classifier_type,
            **metrics,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("f1_micro", ascending=False).reset_index(drop=True)

    logger.info("\n" + "="*60)
    logger.info("多模型对比结果")
    logger.info("="*60)
    logger.info(df_results.to_string(index=True))

    return df_results


def save_model_artifacts(model, vectorizer, mlb, model_name: str = "best_model") -> None:
    """
    保存模型文件

    Args:
        model: 模型
        vectorizer: 向量化器
        mlb: 标签转换器
        model_name: 模型名称
    """
    if vectorizer is None:
        logger.warning("向量化器未训练，跳过保存")
    else:
        vectorizer_path = os.path.join(MODEL_DIR, f"{model_name}_vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"已保存向量化器: {vectorizer_path}")

    if mlb is None:
        logger.warning("标签转换器未训练，跳过保存")
    else:
        mlb_path = os.path.join(MODEL_DIR, f"{model_name}_mlb.pkl")
        with open(mlb_path, "wb") as f:
            pickle.dump(mlb, f)
        logger.info(f"已保存标签转换器: {mlb_path}")

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"已保存模型: {model_path}")


def load_model_artifacts(model_name: str = "best_model") -> Tuple:
    """
    加载模型文件

    Args:
        model_name: 模型名称

    Returns:
        (模型, 向量化器, 标签转换器)
    """
    import os

    vectorizer_path = os.path.join(MODEL_DIR, f"{model_name}_vectorizer.pkl")
    mlb_path = os.path.join(MODEL_DIR, f"{model_name}_mlb.pkl")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")

    logger.info(f"加载向量化器: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info(f"加载标签转换器: {mlb_path}")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    logger.info(f"加载模型: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer, mlb


def split_data(texts: List[str], labels: List[List[str]], test_size: float = 0.2,
               val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    三段划分数据集：训练集/验证集/测试集

    Args:
        texts: 文本列表
        labels: 标签列表
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )

    logger.info(f"数据集划分完成: 训练集 {len(X_train)} / 验证集 {len(X_val)} / 测试集 {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
