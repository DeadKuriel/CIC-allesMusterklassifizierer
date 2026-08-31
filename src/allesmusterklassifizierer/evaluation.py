from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

from .errors import ValidationError


def confusion_tables(
    y_true, y_pred, labels: list[Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unknown = (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(labels)
    if unknown:
        raise ValidationError(
            f"Etiquetas desconocidas en matriz de confusión: {sorted(map(str, unknown))}"
        )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if int(cm.sum()) != len(y_pred):
        raise ValidationError("La suma de la matriz no coincide con las predicciones")
    index = [f"true_{x}" for x in labels]
    columns = [f"pred_{x}" for x in labels]
    raw = pd.DataFrame(cm, index=index, columns=columns)
    rows = raw.div(raw.sum(axis=1).replace(0, 1), axis=0)
    cols = raw.div(raw.sum(axis=0).replace(0, 1), axis=1)
    return raw, rows, cols


def classification_metrics(
    y_true,
    y_pred,
    *,
    labels: list[Any],
    scores=None,
    probabilities=None,
    average="macro",
    positive_class=None,
    zero_division="warn",
) -> dict[str, Any]:
    zd = "warn" if zero_division == "warn" else int(zero_division)
    kwargs = {"average": average, "zero_division": zd}
    if average == "binary":
        kwargs["pos_label"] = positive_class
    cm, _, _ = confusion_tables(y_true, y_pred, labels)
    per = {}
    total: Any = cm.to_numpy().sum()
    for i, lab in enumerate(labels):
        a: Any = cm.to_numpy()
        tp: Any = a[i, i]
        fn: Any = a[i].sum() - tp
        fp: Any = a[:, i].sum() - tp
        tn: Any = total - tp - fn - fp
        per[str(lab)] = {
            "support": int(tp + fn),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        }
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall": recall_score(y_true, y_pred, **kwargs),
        "f1": f1_score(y_true, y_pred, **kwargs),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": per,
        "labels": list(map(str, labels)),
    }
    if probabilities is not None:
        result["log_loss"] = log_loss(y_true, probabilities, labels=labels)
        try:
            result["roc_auc"] = roc_auc_score(
                y_true,
                probabilities[:, 1] if len(labels) == 2 else probabilities,
                multi_class="ovr",
                average=average if average != "binary" else "macro",
            )
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
    if scores is not None and len(labels) == 2:
        binary = (
            np.asarray(y_true)
            == (positive_class if positive_class is not None else labels[1])
        ).astype(int)
        precision, recall, _ = precision_recall_curve(binary, scores)
        result["pr_auc"] = auc(recall, precision)
    return result


def fold_summary(values: list[float], confidence_interval=False) -> dict[str, float]:
    a: Any = np.asarray(values, dtype=float)
    out = {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }
    if confidence_interval and len(a) > 1:
        margin = 1.96 * out["std"] / np.sqrt(len(a))
        out.update(ci95_low=out["mean"] - margin, ci95_high=out["mean"] + margin)
    return out
