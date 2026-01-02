from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _mcc_multiclass_from_cm(cm: np.ndarray) -> float:
    """
    Multiclass MCC (Gorodkin / sklearn-compatible formulation) computed from the confusion matrix.
    Formula:
    MCC = (c*s - sum_k p_k*t_k) / sqrt((s^2 - sum_k p_k^2)*(s^2 - sum_k t_k^2))
    where:
    c = trace(cm)
    s = sum(cm)
    p_k = sum of column k
    t_k = sum of row k
    """
    cm = cm.astype(float)
    s = cm.sum()
    if s == 0:
        return 0.0

    c = np.trace(cm)
    p = cm.sum(axis=0)  # col sums
    t = cm.sum(axis=1)  # row sums

    num = c * s - np.sum(p * t)
    den = np.sqrt((s * s - np.sum(p * p)) * (s * s - np.sum(t * t)))
    return _safe_div(num, den)


def compute_metrics(cm_df: pd.DataFrame, labels: List[Any], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Derived metrics computed from the (raw) confusion matrix.
    Returns:
    - accuracy
    - per_class: precision/recall/specificity/F1 + TP/FN/FP/TN
    - macro: precision/recall/specificity/F1
    - balanced_accuracy (macro recall)
    - mcc (global multiclass)
    """
    # Normalize labels to strings to match the behavior in confusion.py
    labels_s = [str(l) for l in labels]

    cm = cm_df.to_numpy(dtype=float)
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = _safe_div(correct, total)

    per_class: Dict[str, Dict[str, float]] = {}
    precisions = []
    recalls = []
    specificities = []
    f1s = []

    for i, lab in enumerate(labels_s):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP

        precision = _safe_div(TP, (TP + FP))
        recall = _safe_div(TP, (TP + FN))  # sensitivity
        specificity = _safe_div(TN, (TN + FP))
        f1 = _safe_div(2 * precision * recall, (precision + recall))

        per_class[lab] = {
            "tp": float(TP),
            "fn": float(FN),
            "fp": float(FP),
            "tn": float(TN),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
        }

        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_specificity = float(np.mean(specificities)) if specificities else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    balanced_accuracy = macro_recall
    mcc = float(_mcc_multiclass_from_cm(cm))

    # Support: normalize y_true as string
    y_true_s = np.asarray([str(x) for x in y_true], dtype=object)
    support = {lab: int(np.sum(y_true_s == lab)) for lab in labels_s}

    return {
        "accuracy": float(accuracy),
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "sensitivity": macro_recall,
            "specificity": macro_specificity,
            "f1": macro_f1,
        },
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "per_class": per_class,
        "support": support,
        "labels": labels_s,
    }
