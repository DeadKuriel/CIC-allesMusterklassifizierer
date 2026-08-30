from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler

from .errors import SchemaError
from .v1config import ExperimentConfig


class BooleanMapper(BaseEstimator, TransformerMixin):
    def __init__(self, true_values: tuple = (True, 1, "true", "yes"), false_values: tuple = (False, 0, "false", "no")):
        self.true_values = true_values
        self.false_values = false_values

    def fit(self, X: Any, y: Any = None):
        values = set(pd.DataFrame(X).stack().dropna().unique())  # type: ignore[operator]
        ambiguous = set(self.true_values) & set(self.false_values)
        unknown = values - set(self.true_values) - set(self.false_values)
        if ambiguous or unknown:
            raise SchemaError(f"Valores booleanos ambiguos o desconocidos: {sorted(map(str, ambiguous | unknown))}")
        return self

    def transform(self, X: Any):
        frame = pd.DataFrame(X)
        return frame.map(lambda x: np.nan if pd.isna(x) else (1.0 if x in self.true_values else 0.0)).to_numpy()


class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for col in pd.DataFrame(X).columns:
            s = pd.to_datetime(pd.DataFrame(X)[col], errors="raise")
            out.extend([s.dt.year, s.dt.month, s.dt.day, s.dt.dayofweek, s.dt.quarter,
                        np.sin(2 * np.pi * s.dt.month / 12), np.cos(2 * np.pi * s.dt.month / 12)])
        return np.column_stack(out) if out else np.empty((len(X), 0))


def build_preprocessor(cfg: ExperimentConfig) -> ColumnTransformer:
    by_type: dict[str, list[str]] = {}
    columns = {c.name: c for c in cfg.columns if c.role == "feature"}
    for c in columns.values():
        by_type.setdefault(c.semantic_type, []).append(c.name)
    transformers = []
    p = cfg.preprocessing
    scale = {"none": "passthrough", "standard": StandardScaler(), "minmax": MinMaxScaler(),
             "robust": RobustScaler(), "normalize": Normalizer(), "power": PowerTransformer()}[p.scale]
    if numeric := by_type.get("numeric"):
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy=p.numeric_imputation,
            fill_value=p.numeric_constant, add_indicator=p.add_missing_indicators)), ("scale", scale)]), numeric))
    if nominal := by_type.get("nominal"):
        strategy = "constant" if p.categorical_imputation in {"constant", "missing_category"} else "most_frequent"
        transformers.append(("nominal", Pipeline([("imputer", SimpleImputer(strategy=strategy,
            fill_value=p.categorical_constant, add_indicator=p.add_missing_indicators)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=p.min_category_frequency))]), nominal))
    if ordinal := by_type.get("ordinal"):
        categories = [columns[name].order for name in ordinal]
        transformers.append(("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1))]), ordinal))
    if boolean := by_type.get("boolean"):
        c = columns[boolean[0]]
        transformers.append(("boolean", Pipeline([("mapper", BooleanMapper(tuple(c.true_values), tuple(c.false_values))),
            ("imputer", SimpleImputer(strategy="most_frequent"))]), boolean))
    if dates := by_type.get("datetime"):
        transformers.append(("datetime", Pipeline([("date", DateFeatures()), ("imputer", SimpleImputer(strategy="median")),
                                                    ("scale", scale)]), dates))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_pipeline(cfg: ExperimentConfig, estimator):
    from imblearn.pipeline import Pipeline as ImbPipeline
    steps = [("preprocess", build_preprocessor(cfg))]
    strategy = cfg.balance.strategy
    if strategy != "none":
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
        from imblearn.under_sampling import RandomUnderSampler
        samplers = {"random_under": RandomUnderSampler, "random_over": RandomOverSampler,
                    "smote": SMOTE, "smotenc": SMOTENC}
        params = {"random_state": cfg.seed, **cfg.balance.params}
        if strategy == "smotenc" and "categorical_features" not in params:
            raise SchemaError("SMOTENC requiere categorical_features explícito tras el preprocesamiento")
        steps.append(("sampler", samplers[strategy](**params)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)
