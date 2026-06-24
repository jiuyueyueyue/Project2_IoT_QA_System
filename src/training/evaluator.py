"""
src/training/evaluator.py
==========================
多标签分类评估器 — 计算全面的多标签评估指标。

变更说明（vs 旧 utils/train_eval.py evaluate_model()）:
    - 旧: 仅 5 个聚合指标，无 per-class 分解
    - 新: 聚合 + per-class + Subset Accuracy + Exact Match Ratio
    - 原因: 多标签场景下单看 micro/macro 不够，需要 per-class 定位薄弱类别

评估指标:
    聚合指标:
        - Subset Accuracy (Exact Match Ratio): 完全正确的样本占比
        - Hamming Loss: 汉明损失（越小越好）
        - F1-micro / F1-macro / F1-weighted
        - Jaccard-micro / Jaccard-macro
        - Precision-micro / Recall-micro

    逐类指标 (Per-Class):
        - Precision / Recall / F1-score for each label
        - Support (样本数)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """多标签分类评估器。

    提供聚合评估与逐类评估两种视角，帮助诊断模型在各类别上的表现差异。

    Example:
        evaluator = Evaluator(label_names=mlb.classes_)
        metrics = evaluator.evaluate(y_test_mat, y_pred_mat)
        print(evaluator.report(metrics))
    """

    def __init__(self, label_names: Optional[List[str]] = None) -> None:
        """初始化评估器。

        Args:
            label_names: 标签名称列表（用于 per-class 报告，如 mlb.classes_）
        """
        self.label_names = label_names or []

    # ============================================================
    # 核心评估方法
    # ============================================================

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """计算完整的评估指标集合。

        Args:
            y_true: 真实标签矩阵 (n_samples, n_labels)，值为 0/1
            y_pred: 预测标签矩阵 (n_samples, n_labels)，值为 0/1
            y_proba: 预测概率矩阵 (n_samples, n_labels)，可选

        Returns:
            指标字典，包含:
                - subset_accuracy: 精确匹配率
                - hamming_loss: 汉明损失
                - f1_micro / f1_macro / f1_weighted
                - precision_micro / recall_micro
                - jaccard_micro / jaccard_macro
                - per_class: 逐类指标 DataFrame
                - per_class_dict: 逐类指标字典
                - zero_division: 全零预测的类别数
        """
        n_samples, n_labels = y_true.shape

        # --- 聚合指标 ---
        metrics: Dict = {
            "n_samples": n_samples,
            "n_labels": n_labels,
            "subset_accuracy": accuracy_score(y_true, y_pred),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
            "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        }

        # --- 逐类指标 ---
        per_class = self._per_class_metrics(y_true, y_pred)
        metrics["per_class"] = per_class
        metrics["per_class_dict"] = per_class.to_dict(orient="index")

        # --- 全零预测类别数 ---
        zero_pred_labels = int((y_pred.sum(axis=0) == 0).sum())
        metrics["zero_division_labels"] = zero_pred_labels
        if zero_pred_labels > 0:
            logger.warning(f"存在 {zero_pred_labels} 个类别预测全为 0（zero_division）")

        return metrics

    # ============================================================
    # 逐类指标
    # ============================================================

    def _per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """计算每个标签的 Precision / Recall / F1 / Support。

        Args:
            y_true: 真实标签矩阵
            y_pred: 预测标签矩阵

        Returns:
            逐类指标 DataFrame
        """
        rows: List[Dict] = []
        n_labels = y_true.shape[1]
        names = self.label_names if len(self.label_names) == n_labels else [
            f"label_{i}" for i in range(n_labels)
        ]

        for i, name in enumerate(names):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            rows.append({
                "label": name,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "support_true": int(yt.sum()),
                "support_pred": int(yp.sum()),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
        return df

    # ============================================================
    # 分类报告
    # ============================================================

    @staticmethod
    def classification_report_text(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """生成 sklearn 格式的分类报告（多标签版）。

        注意: sklearn 的 classification_report 对多标签使用 precision_recall_fscore_support
        每个标签独立计算。

        Args:
            y_true: 真实标签矩阵
            y_pred: 预测标签矩阵
            target_names: 标签名称

        Returns:
            分类报告字符串
        """
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0,
        )

    # ============================================================
    # 格式化输出
    # ============================================================

    def report(self, metrics: Dict) -> str:
        """生成人类可读的评估报告。

        Args:
            metrics: evaluate() 返回的指标字典

        Returns:
            格式化报告字符串
        """
        lines = [
            "=" * 70,
            "多标签分类评估报告",
            "=" * 70,
            f"样本数:       {metrics['n_samples']}",
            f"标签数:       {metrics['n_labels']}",
            f"零预测标签数: {metrics['zero_division_labels']}",
            "-" * 70,
            "聚合指标:",
            f"  Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.4f}",
            f"  Hamming Loss:                  {metrics['hamming_loss']:.4f}",
            f"  F1-micro:                      {metrics['f1_micro']:.4f}",
            f"  F1-macro:                      {metrics['f1_macro']:.4f}",
            f"  F1-weighted:                   {metrics['f1_weighted']:.4f}",
            f"  Precision-micro:               {metrics['precision_micro']:.4f}",
            f"  Recall-micro:                  {metrics['recall_micro']:.4f}",
            f"  Jaccard-micro:                 {metrics['jaccard_micro']:.4f}",
            f"  Jaccard-macro:                 {metrics['jaccard_macro']:.4f}",
            "-" * 70,
            "逐类指标 (Per-Class):",
        ]

        df: pd.DataFrame = metrics["per_class"]
        for _, row in df.iterrows():
            lines.append(
                f"  {row['label']:<12s} "
                f"P={row['precision']:.4f} "
                f"R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} "
                f"(支持: T={int(row['support_true'])} P={int(row['support_pred'])})"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    # ============================================================
    # 模型对比（静态方法，可独立使用）
    # ============================================================

    @staticmethod
    def compare(
        metrics_list: List[Dict],
        model_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """对比多个模型的聚合指标。

        Args:
            metrics_list: 各模型 evaluate() 返回的指标字典列表
            model_names: 对应的模型名称列表

        Returns:
            对比结果 DataFrame
        """
        keys = [
            "subset_accuracy", "hamming_loss",
            "f1_micro", "f1_macro", "f1_weighted",
            "jaccard_micro", "jaccard_macro",
        ]
        rows = []
        for i, m in enumerate(metrics_list):
            name = model_names[i] if model_names else f"model_{i}"
            row = {"model": name}
            for k in keys:
                row[k] = m.get(k, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("f1_micro", ascending=False).reset_index(drop=True)
        return df
