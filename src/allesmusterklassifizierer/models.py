from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .errors import ClassifierError


@dataclass(frozen=True)
class ModelSpec:
    constructor: Callable[..., BaseEstimator]
    defaults: dict[str, Any]
    probability: bool
    class_weight: bool
    scaling_recommended: bool
    search_space: dict[str, Any]


REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(LogisticRegression, {"max_iter": 1000}, True, True, True, {"C": [0.1, 1, 10]}),
    "knn": ModelSpec(KNeighborsClassifier, {}, True, False, True, {"n_neighbors": [1, 3, 5, 9]}),
    "centroid": ModelSpec(NearestCentroid, {}, False, False, True, {}),
    "decision_tree": ModelSpec(DecisionTreeClassifier, {}, True, True, False, {"max_depth": [None, 3, 8]}),
    "random_forest": ModelSpec(RandomForestClassifier, {"n_estimators": 200}, True, True, False, {"max_depth": [None, 5, 10]}),
    "extra_trees": ModelSpec(ExtraTreesClassifier, {"n_estimators": 200}, True, True, False, {"max_depth": [None, 5, 10]}),
    "linear_svm": ModelSpec(LinearSVC, {}, False, True, True, {"C": [0.1, 1, 10]}),
    "kernel_svm": ModelSpec(SVC, {"probability": True}, True, True, True, {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
    "gaussian_nb": ModelSpec(GaussianNB, {}, True, False, False, {}),
    "gradient_boosting": ModelSpec(GradientBoostingClassifier, {}, True, False, False, {"n_estimators": [50, 100]}),
    "hist_gradient_boosting": ModelSpec(HistGradientBoostingClassifier, {}, True, True, False, {"max_iter": [100, 200]}),
    "adaboost": ModelSpec(AdaBoostClassifier, {}, True, False, False, {"n_estimators": [50, 100]}),
    "mlp": ModelSpec(MLPClassifier, {"max_iter": 500}, True, False, True, {"hidden_layer_sizes": [(50,), (100,)]}),
}


ALIASES = {"euclidiano": "centroid", "euclidean_centroid": "centroid", "svm": "kernel_svm"}
DISTANCE_ALIASES = {"euclidean": "euclidean", "manhattan": "manhattan", "cityblock": "manhattan",
                    "chebyshev": "chebyshev", "chessboard": "chebyshev", "minkowski": "minkowski"}


def normalize_distance(value: Any) -> tuple[str, float]:
    if isinstance(value, dict):
        name, p = value.get("type"), value.get("params", {}).get("p")
    else:
        name, p = value, None
    if name not in DISTANCE_ALIASES:
        raise ClassifierError(f"Distancia no soportada: {name}. Usa {sorted(DISTANCE_ALIASES)}")
    metric = DISTANCE_ALIASES[name]
    if metric == "minkowski":
        if not isinstance(p, (int, float)) or p <= 0:
            raise ClassifierError("Minkowski requiere p numérico > 0")
        return metric, float(p)
    return metric, 2.0


def create_model(name: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    name = ALIASES.get(name, name)
    if name not in REGISTRY:
        raise ClassifierError(f"Modelo desconocido '{name}'. Disponibles: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    values = {**spec.defaults, **params}
    if name == "knn" and "distance" in values:
        metric, p = normalize_distance(values.pop("distance"))
        values["metric"] = metric
        if metric == "minkowski": values["p"] = p
    if "random_state" in spec.constructor().get_params() and "random_state" not in values:
        values["random_state"] = seed
    try:
        return spec.constructor(**values)
    except TypeError as exc:
        raise ClassifierError(f"Parámetros inválidos para {name}: {exc}") from exc


class AcademicCentroid(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        import numpy as np
        self.classes_ = np.unique(y); self.centroids_ = np.stack([np.mean(X[np.asarray(y) == c], axis=0) for c in self.classes_]); return self
    def predict(self, X):
        import numpy as np
        return self.classes_[np.argmin(((np.asarray(X)[:, None] - self.centroids_) ** 2).sum(axis=2), axis=1)]
