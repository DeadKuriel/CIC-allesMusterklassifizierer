from __future__ import annotations
from typing import Hashable, Dict, List

import numpy as np


class EuclideanClassifier:
    """
    Clasificador Euclidiano simple:
    - Calcula el centroide (media) de cada clase en el espacio de características.
    - Predice asignando cada patrón a la clase con centroide más cercano
      (distancia euclidiana).
    """

    def __init__(self):
        # Diccionario: clase -> vector centroide (np.ndarray)
        self.class_means_: Dict[Hashable, np.ndarray] | None = None
        # Arreglo con las clases en el orden usado internamente
        self.classes_: np.ndarray | None = None
        # Nombres de características si X es un DataFrame
        self.feature_names_: List[str] | None = None

    def fit(self, X, y):
        """
        Ajusta el clasificador calculando el centroide de cada clase.
        X: matriz o DataFrame (n_muestras x n_características)
        y: vector de etiquetas (n_muestras)
        """
        # Guardar nombres de columnas si X es DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
            X_values = X.to_numpy()
        else:
            self.feature_names_ = None
            X_values = np.asarray(X)

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        self.classes_ = classes
        self.class_means_ = {}

        for c in classes:
            self.class_means_[c] = X_values[y_arr == c].mean(axis=0)

        return self

    def predict(self, X):
        """
        Predice la clase de cada patrón en X usando distancia euclidiana
        a los centroides.
        """
        if self.class_means_ is None or self.classes_ is None:
            raise RuntimeError("Debe llamar a fit antes de predict.")

        if hasattr(X, "columns"):
            X_values = X.to_numpy()
        else:
            X_values = np.asarray(X)

        # Matriz de centroides: K x D
        centroids = np.stack(
            [self.class_means_[c] for c in self.classes_],
            axis=0
        )  # (K, D)

        # Distancias: (N, 1, D) - (1, K, D) -> (N, K, D) -> norma en eje 2: (N, K)
        diff = X_values[:, None, :] - centroids[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (N, K)

        idx_min = np.argmin(dists, axis=1)
        preds = self.classes_[idx_min]
        return preds

    def distances_to_centroids(self, X):
        """
        Calcula la matriz de distancias de cada patrón en X a cada centroide.

        Devuelve un diccionario:
            {
                "distances": ndarray de forma (n_muestras, n_clases),
                "classes": lista de clases en el mismo orden que las columnas
            }
        """
        if self.class_means_ is None or self.classes_ is None:
            raise RuntimeError(
                "Debe llamar a fit antes de distances_to_centroids."
            )

        if hasattr(X, "columns"):
            X_values = X.to_numpy()
        else:
            X_values = np.asarray(X)

        centroids = np.stack(
            [self.class_means_[c] for c in self.classes_],
            axis=0
        )  # (K, D)
        diff = X_values[:, None, :] - centroids[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (N, K)

        return {
            "distances": dists,
            "classes": list(self.classes_),
        }
