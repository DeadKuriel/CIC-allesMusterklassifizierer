from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..distances.metrics import DistanceName, pairwise_distance
from ..errors import ClassifierError


@dataclass
class KNNClassifier:
    k: int = 1
    distance: DistanceName = "euclidean"
    X_train_: np.ndarray | None = None
    y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ClassifierError("X debe ser 2D.")
        if y.shape[0] != X.shape[0]:
            raise ClassifierError("len(y) debe coincidir con n_rows(X).")
        if not isinstance(self.k, int) or self.k <= 0:
            raise ClassifierError("k debe ser entero positivo.")
        if self.k > X.shape[0]:
            raise ClassifierError(f"k={self.k} no puede ser mayor que n_train={X.shape[0]}.")

        self.X_train_ = X
        self.y_train_ = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise ClassifierError("El clasificador no está entrenado. Llama fit() primero.")
        if X.ndim != 2:
            raise ClassifierError("X debe ser 2D.")

        preds = np.empty(X.shape[0], dtype=object)
        for i in range(X.shape[0]):
            preds[i] = self._predict_one(X[i])
        return preds

    def _predict_one(self, x: np.ndarray) -> Any:
        Xtr = self.X_train_
        ytr = self.y_train_
        assert Xtr is not None and ytr is not None

        dists = pairwise_distance(Xtr, x, self.distance)
        nn_idx = np.argpartition(dists, self.k - 1)[: self.k]
        # sort the chosen neighbors by distance for stable tie handling
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]

        labels = ytr[nn_idx]
        # vote counts
        unique, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        top = unique[counts == max_count]

        if top.size == 1:
            return top[0]

        # tie-break 1: choose class with smaller sum of distances among its neighbors
        sum_dist = {}
        for c in top:
            sum_dist[c] = float(np.sum(dists[nn_idx][labels == c]))
        best_sum = min(sum_dist.values())
        top2 = [c for c, sd in sum_dist.items() if sd == best_sum]
        if len(top2) == 1:
            return top2[0]

        # tie-break 2: deterministic by string ordering
        return sorted(top2, key=lambda z: str(z))[0]
