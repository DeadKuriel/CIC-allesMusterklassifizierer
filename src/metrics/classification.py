from __future__ import annotations
import numpy as np

def accuracy(y_true, y_pred) -> float:
    """
    Calcula el Accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()


def imbalance_ratio(y) -> float:
    """
    Calcula el Imbalance Ratio (IR) de un vector de clases.
    IR = (tamaño de la clase mayoritaria) / (tamaño de la clase minoritaria)
    Si solo hay una clase, devolvemos 1.0 (no hay desbalance en términos relativos).
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    classes, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        # Solo una clase presente
        return 1.0

    max_count = counts.max()
    min_count = counts.min()

    return max_count / min_count
