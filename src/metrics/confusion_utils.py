from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion_matrices(y_true, y_pred, classes):
    """
    Devuelve:
      - cm_cruda
      - cm_filas (normalizada por filas)
      - cm_columnas (normalizada por columnas)
    todas como np.array.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    cm_rows = cm.astype(float)
    row_sums = cm_rows.sum(axis=1, keepdims=True)
    cm_rows = np.divide(cm_rows, row_sums, where=row_sums != 0)

    cm_cols = cm.astype(float)
    col_sums = cm_cols.sum(axis=0, keepdims=True)
    cm_cols = np.divide(cm_cols, col_sums, where=col_sums != 0)

    return cm, cm_rows, cm_cols


def binary_metrics_from_cm(cm):
    """
    La matriz cm viene en formato:
        [[TP, FN],
         [FP, TN]]
    """

    import numpy as np

    TP, FN = cm[0]
    FP, TN = cm[1]

    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    sensitivity = safe_div(TP, TP + FN)
    specificity = safe_div(TN, TN + FP)
    precision = safe_div(TP, TP + FP)

    if np.isnan(precision) or np.isnan(sensitivity) or (precision + sensitivity == 0):
        f1 = np.nan
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)

    if (TP+FP)==0 or (TP+FN)==0 or (TN+FP)==0 or (TN+FN)==0:
        mcc = np.nan
    else:
        mcc = ((TP * TN) - (FP * FN)) / np.sqrt(
            (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)
        )

    bal_acc = np.nan
    if not np.isnan(sensitivity) and not np.isnan(specificity):
        bal_acc = (sensitivity + specificity) / 2

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "f1": f1,
        "mcc": mcc
    }
