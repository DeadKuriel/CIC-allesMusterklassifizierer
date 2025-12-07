from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseClassifier

class EuclideanClassifier(BaseClassifier):
    """
    Clasificador euclidiano clásico:
    - Calcula el vector media (centroide) de cada clase en el espacio de características.
    - Clasifica un patrón según la clase con media más cercana (distancia euclidiana).
    """
    def __init__(self):
        self.class_means_: dict = {}

    def fit(self, X: pd.DataFrame, y):
        X_np = X.to_numpy()
        y_np = np.asarray(y)

        classes = np.unique(y_np)
        self.class_means_ = {}

        for c in classes:
            self.class_means_[c] = X_np[y_np == c].mean(axis=0)

        return self

    def predict(self, X: pd.DataFrame):
        X_np = X.to_numpy()
        preds = []

        for x in X_np:
            best_class = None
            best_dist = None
            for c, mu_c in self.class_means_.items():
                dist = np.linalg.norm(x - mu_c)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_class = c
            preds.append(best_class)

        return np.array(preds)
