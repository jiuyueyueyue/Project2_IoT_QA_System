"""
src/training/trainer.py
========================
模型训练器 — 封装完整训练流水线。

变更说明（vs 旧 train.py + utils/train_eval.py）:
    - 旧: 函数式风格，全局 import config 参数
    - 新: Trainer 类封装，依赖注入 Settings，所有参数可覆盖
    - 原因: 企业级封装使得训练流程可测试、可配置、可继承

训练流水线:
    1. 数据划分 (train/val/test)
    2. 特征提取 (TF-IDF)
    3. 标签编码 (MultiLabelBinarizer)
    4. 模型训练 (OneVsRest + 基分类器)
    5. 可选超参数调优 (GridSearchCV)
    6. 模型评估
    7. 模型持久化
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from config.settings import CLASSIFIER_REGISTRY, DEFAULT_CLASSIFIER, settings
from src.training.evaluator import Evaluator
from src.utils.exceptions import TrainingError

logger = logging.getLogger(__name__)


class Trainer:
    """多标签分类训练器。

    封装从数据划分到模型持久化的完整训练生命周期。

    Attributes:
        model: 训练后的分类器（OneVsRestClassifier）
        mlb: 标签二值化器（MultiLabelBinarizer）
        classifier_type: 当前使用的分类器类型 key
        is_trained: 模型是否已训练
        metrics: 评估指标（训练完成后填充）

    Example:
        trainer = Trainer(classifier_type="logistic_regression")
        trainer.train(X_train_vec, y_train, X_test_vec, y_test)
        trainer.save_artifacts(vectorizer)
    """

    def __init__(
        self,
        classifier_type: str = DEFAULT_CLASSIFIER,
    ) -> None:
        """初始化训练器。

        Args:
            classifier_type: 分类器注册表中的 key（logistic_regression / random_forest / svm）

        Raises:
            ValueError: 分类器类型不存在
        """
        if classifier_type not in CLASSIFIER_REGISTRY:
            raise ValueError(
                f"未知分类器类型: {classifier_type}，可选: {list(CLASSIFIER_REGISTRY.keys())}"
            )

        self.classifier_type = classifier_type
        self._config = CLASSIFIER_REGISTRY[classifier_type]
        self.model: Optional[OneVsRestClassifier] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.is_trained = False
        self.metrics: Dict = {}
        self.evaluator: Optional[Evaluator] = None

    # ============================================================
    # 数据划分
    # ============================================================

    @staticmethod
    def split_data(
        texts: List[str],
        labels: List[List[str]],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[List[str], List[str], List[str], List[List[str]], List[List[str]], List[List[str]]]:
        """三段划分: 训练集 / 验证集 / 测试集。

        Args:
            texts: 文本列表
            labels: 标签列表
            test_size: 测试集占比
            val_size: 验证集占 (1 - test_size) 的比例
            random_state: 随机种子

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Step 1: 分出测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )

        # Step 2: 从剩余中分出验证集
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
        )

        logger.info(
            f"数据集划分: 训练集 {len(X_train)} / 验证集 {len(X_val)} / 测试集 {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ============================================================
    # 标签编码
    # ============================================================

    def fit_binarizer(self, labels: List[List[str]]) -> MultiLabelBinarizer:
        """拟合 MultiLabelBinarizer。

        Args:
            labels: 标签列表

        Returns:
            拟合好的 MultiLabelBinarizer 实例
        """
        logger.info("拟合 MultiLabelBinarizer ...")
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(labels)
        logger.info(f"标签类别: {list(self.mlb.classes_)} ({len(self.mlb.classes_)} 类)")
        return self.mlb

    # ============================================================
    # 模型创建
    # ============================================================

    def _create_base_classifier(self):
        """根据分类器类型创建基分类器实例。"""
        model_name = self._config["model"]
        params = self._config["params"]

        if model_name == "LogisticRegression":
            return LogisticRegression(**params)
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(**params)
        elif model_name == "SVC":
            return SVC(**params)
        else:
            raise ValueError(f"不支持的基分类器: {model_name}")

    def create_classifier(self) -> OneVsRestClassifier:
        """创建 OneVsRest 包装的分类器。

        Returns:
            OneVsRestClassifier 实例
        """
        base = self._create_base_classifier()
        return OneVsRestClassifier(base)

    # ============================================================
    # 训练
    # ============================================================

    def train(
        self,
        X_train,
        y_train_mat: np.ndarray,
        X_val=None,
        y_val_mat: Optional[np.ndarray] = None,
    ) -> OneVsRestClassifier:
        """训练/拟合分类器。

        Args:
            X_train: 训练特征矩阵
            y_train_mat: 训练标签矩阵
            X_val: 验证特征矩阵（可选，仅用于日志）
            y_val_mat: 验证标签矩阵（可选，仅用于日志）

        Returns:
            训练好的 OneVsRestClassifier

        Raises:
            TrainingError: 训练失败
        """
        logger.info(f"训练分类器: {self.classifier_type} ({self._config['name']})")

        try:
            self.model = self.create_classifier()
            self.model.fit(X_train, y_train_mat)
            self.is_trained = True
            logger.info("分类器训练完成")

            # 若提供了验证集，进行快速验证
            if X_val is not None and y_val_mat is not None:
                val_pred = self.model.predict(X_val)
                from sklearn.metrics import f1_score
                val_f1 = f1_score(y_val_mat, val_pred, average="micro", zero_division=0)
                logger.info(f"验证集 F1-micro: {val_f1:.4f}")

            return self.model

        except Exception as e:
            raise TrainingError(str(e)) from e

    def grid_search_train(
        self,
        X_train,
        y_train_mat: np.ndarray,
        cv: int = 3,
        n_jobs: int = -1,
    ) -> Tuple[OneVsRestClassifier, Dict]:
        """使用 GridSearchCV 进行超参数调优并训练。

        Args:
            X_train: 训练特征矩阵
            y_train_mat: 训练标签矩阵
            cv: 交叉验证折数
            n_jobs: 并行数（-1 表示全部 CPU）

        Returns:
            (最佳模型, 调优结果字典)

        Raises:
            TrainingError: 超参数搜索失败
        """
        grid_params = self._config.get("grid_params")
        if not grid_params:
            raise TrainingError(f"分类器 {self.classifier_type} 未定义 grid_params")

        logger.info(f"开始 GridSearchCV 调优: {self.classifier_type}")
        logger.info(f"搜索空间: {grid_params}")

        try:
            model = self.create_classifier()
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid_params,
                cv=cv,
                scoring="f1_micro",
                n_jobs=n_jobs,
                verbose=1,
            )
            grid_search.fit(X_train, y_train_mat)

            self.model = grid_search.best_estimator_
            self.is_trained = True

            result = {
                "best_params": grid_search.best_params_,
                "best_score": float(grid_search.best_score_),
            }

            logger.info(f"GridSearchCV 完成 — 最佳参数: {result['best_params']}")
            logger.info(f"最佳 CV F1-micro: {result['best_score']:.4f}")

            return self.model, result

        except Exception as e:
            raise TrainingError(f"GridSearchCV 失败: {e}") from e

    # ============================================================
    # 评估
    # ============================================================

    def evaluate(
        self,
        X_test,
        y_test_mat: np.ndarray,
    ) -> Dict:
        """在测试集上评估模型。

        Args:
            X_test: 测试特征矩阵
            y_test_mat: 测试标签矩阵

        Returns:
            评估指标字典

        Raises:
            TrainingError: 模型未训练
        """
        if not self.is_trained or self.model is None:
            raise TrainingError("模型未训练，无法评估")

        y_pred = self.model.predict(X_test)

        # 尝试获取概率（部分模型支持 predict_proba）
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_test)
            except Exception:
                pass

        self.evaluator = Evaluator(label_names=list(self.mlb.classes_) if self.mlb else None)
        self.metrics = self.evaluator.evaluate(y_test_mat, y_pred, y_proba)

        logger.info(self.evaluator.report(self.metrics))
        return self.metrics

    # ============================================================
    # 模型对比
    # ============================================================

    @staticmethod
    def compare_models(
        X_train,
        y_train_mat: np.ndarray,
        X_test,
        y_test_mat: np.ndarray,
        classifier_types: Optional[List[str]] = None,
        use_grid_search: bool = False,
        label_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """多模型对比实验。

        Args:
            X_train: 训练特征
            y_train_mat: 训练标签
            X_test: 测试特征
            y_test_mat: 测试标签
            classifier_types: 参与对比的分类器列表（默认全部）
            use_grid_search: 是否启用网格搜索
            label_names: 标签名列表

        Returns:
            (对比结果 DataFrame, 各模型指标列表)
        """
        import pandas as pd
        types = classifier_types or list(CLASSIFIER_REGISTRY.keys())
        all_metrics = []

        for ct in types:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练模型: {ct} ({CLASSIFIER_REGISTRY[ct]['name']})")
            logger.info(f"{'='*60}")

            trainer = Trainer(classifier_type=ct)
            if use_grid_search:
                trainer.grid_search_train(X_train, y_train_mat)
            else:
                trainer.train(X_train, y_train_mat)

            metrics = trainer.evaluate(X_test, y_test_mat)
            metrics["_type"] = ct
            metrics["_name"] = CLASSIFIER_REGISTRY[ct]["name"]
            all_metrics.append(metrics)

        df = Evaluator.compare(
            all_metrics,
            model_names=[m["_name"] for m in all_metrics],
        )
        logger.info("\n" + df.to_string(index=False))
        return df, all_metrics

    # ============================================================
    # 持久化
    # ============================================================

    def save_artifacts(
        self,
        vectorizer: Any,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        mlb_path: Optional[Path] = None,
    ) -> None:
        """保存模型文件、向量化器、标签转换器。

        Args:
            vectorizer: TF-IDF 向量化器实例
            model_path: 模型保存路径
            vectorizer_path: 向量化器保存路径
            mlb_path: 标签转换器保存路径

        Raises:
            TrainingError: 模型或标签转换器未训练
        """
        if not self.is_trained:
            raise TrainingError("模型未训练，无法保存")

        model_path = model_path or settings.model_path
        vectorizer_path = vectorizer_path or settings.vectorizer_path
        mlb_path = mlb_path or settings.mlb_path

        # 保存模型
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存: {model_path}")

        # 保存向量化器
        if vectorizer is not None:
            if hasattr(vectorizer, "save"):
                vectorizer.save(vectorizer_path)
            else:
                with open(vectorizer_path, "wb") as f:
                    pickle.dump(vectorizer, f)
                logger.info(f"向量化器已保存: {vectorizer_path}")

        # 保存标签转换器
        if self.mlb is not None:
            mlb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mlb_path, "wb") as f:
                pickle.dump(self.mlb, f)
            logger.info(f"标签转换器已保存: {mlb_path}")

    @classmethod
    def load_artifacts(
        cls,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        mlb_path: Optional[Path] = None,
    ) -> Tuple[Any, Any, MultiLabelBinarizer]:
        """加载模型文件、向量化器、标签转换器。

        Args:
            model_path: 模型文件路径
            vectorizer_path: 向量化器文件路径
            mlb_path: 标签转换器文件路径

        Returns:
            (模型, 向量化器, MultiLabelBinarizer)
        """
        model_path = model_path or settings.model_path
        vectorizer_path = vectorizer_path or settings.vectorizer_path
        mlb_path = mlb_path or settings.mlb_path

        logger.info(f"加载模型: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"加载向量化器: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        logger.info(f"加载标签转换器: {mlb_path}")
        with open(mlb_path, "rb") as f:
            mlb = pickle.load(f)

        logger.info("模型文件加载完成")
        return model, vectorizer, mlb
