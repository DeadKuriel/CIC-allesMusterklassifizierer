from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ..errors import ClassifierError


@dataclass
class EuclideanCentroidClassifier:
    """
    Nearest Centroid / Minimum Distance to Class Centroids.
    Distance: Fixed Euclidean (L2) metric, not configurable.
    """
    centroids_: Dict[Any, np.ndarray] | None = None
    classes_: List[Any] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ClassifierError("X debe ser 2D.")
        if y.shape[0] != X.shape[0]:
            raise ClassifierError("len(y) debe coincidir con n_rows(X).")

        classes = sorted(list(np.unique(y)), key=lambda z: str(z))
        centroids: Dict[Any, np.ndarray] = {}
        for c in classes:
            Xc = X[y == c]
            if Xc.size == 0:
                raise ClassifierError(f"Clase vacía detectada: {c}")
            centroids[c] = np.mean(Xc, axis=0)

        self.centroids_ = centroids
        self.classes_ = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.classes_ is None:
            raise ClassifierError("El clasificador no está entrenado. Llama fit() primero.")
        if X.ndim != 2:
            raise ClassifierError("X debe ser 2D.")

        C = len(self.classes_)
        n = X.shape[0]
        preds = np.empty(n, dtype=object)

        # Stack centroids for vectorized distance computation
        centroid_mat = np.stack([self.centroids_[c] for c in self.classes_], axis=0)  # (C, d)

        for i in range(n):
            x = X[i]  # (d,)
            diff = centroid_mat - x  # (C, d)
            dists = np.sqrt(np.sum(diff * diff, axis=1))  # L2 fixed
            j = int(np.argmin(dists))
            preds[i] = self.classes_[j]

        return preds
