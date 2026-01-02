from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, labels: List[Any]) -> pd.DataFrame:
    """
    Confusion matrix implementation without relying on sklearn (prevents 'binary vs unknown' target issues).
    - All labels are internally converted to strings to guarantee consistency.
    - The `labels` parameter specifies the class order (also converted to strings).
    """
    y_true_s = np.asarray([str(x) for x in y_true], dtype=object)
    y_pred_s = np.asarray([str(x) for x in y_pred], dtype=object)
    labels_s = [str(l) for l in labels]

    label_to_i = {lab: i for i, lab in enumerate(labels_s)}
    cm = np.zeros((len(labels_s), len(labels_s)), dtype=int)

    # Just if in labels
    for yt, yp in zip(y_true_s, y_pred_s):
        if yt in label_to_i and yp in label_to_i:
            cm[label_to_i[yt], label_to_i[yp]] += 1

    df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels_s],
        columns=[f"pred_{l}" for l in labels_s],
    )
    return df


def normalize_by_rows(cm_df: pd.DataFrame) -> pd.DataFrame:
    cm = cm_df.to_numpy(dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    norm = cm / row_sums
    return pd.DataFrame(norm, index=cm_df.index, columns=cm_df.columns)


def normalize_by_columns(cm_df: pd.DataFrame) -> pd.DataFrame:
    cm = cm_df.to_numpy(dtype=float)
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0.0] = 1.0
    norm = cm / col_sums
    return pd.DataFrame(norm, index=cm_df.index, columns=cm_df.columns)
