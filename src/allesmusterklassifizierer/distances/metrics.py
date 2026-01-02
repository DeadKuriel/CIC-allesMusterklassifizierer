from __future__ import annotations

from typing import Literal

import numpy as np

DistanceName = Literal["euclidean", "manhattan", "chebyshev"]


def pairwise_distance(X_train: np.ndarray, x: np.ndarray, metric: DistanceName) -> np.ndarray:
    """
    Distance between each row of X_train and a vector x.
    Returns a vector of shape (n_train,).
    """
    diff = X_train - x  # broadcast
    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=1))
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=1)
    if metric == "chebyshev":
        return np.max(np.abs(diff), axis=1)
    raise ValueError(f"Distancia no soportada: {metric}")
