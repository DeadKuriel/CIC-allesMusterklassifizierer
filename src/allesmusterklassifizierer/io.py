from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .errors import DataError


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_name: str
    index: np.ndarray  # original row index
    classes: List[Any]  # sorted unique labels


def _ensure_exists(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise DataError(f"No existe el archivo: {path}")
    if p.suffix.lower() != ".csv":
        raise DataError(f"Se esperaba CSV (.csv). Recibido: {path}")
    return p


def load_csv_dataset(
    csv_path: str,
    target_col: str = "class",
    drop_cols: Optional[List[str]] = None,
) -> Dataset:
    p = _ensure_exists(csv_path)
    df = pd.read_csv(p)

    if target_col not in df.columns:
        raise DataError(f"Columna target '{target_col}' no existe en {csv_path}. Columnas: {list(df.columns)}")

    drop_cols = drop_cols or []
    for c in drop_cols:
        if c not in df.columns:
            raise DataError(f"drop_cols incluye '{c}', pero no existe en el dataset. Columnas: {list(df.columns)}")

    # separate y
    y = df[target_col].to_numpy()
    X_df = df.drop(columns=[target_col] + drop_cols)

    # ensure numeric
    non_numeric = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise DataError(
            "Features no numéricos detectados: "
            f"{non_numeric}. Solución: elimina/convierte esas columnas en el CSV o usa drop_cols."
        )

    X = X_df.to_numpy(dtype=float, copy=True)
    if X.ndim != 2:
        raise DataError("X debe ser una matriz 2D (n_samples, n_features).")

    if len(y) != X.shape[0]:
        raise DataError("Inconsistencia: len(y) != n_rows(X).")
    if X.shape[1] == 0:
        raise DataError("El dataset no contiene características.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise DataError("Features con NaN o infinitos; usa el flujo v1 para imputación trazable.")
    if pd.isna(y).any() or pd.Series(y).nunique() < 2:
        raise DataError("El objetivo no puede ser nulo y debe contener al menos dos clases.")

    # classes stable ordering
    classes = sorted(pd.unique(pd.Series(y)).tolist(), key=lambda z: str(z))
    idx = np.arange(X.shape[0], dtype=int)

    return Dataset(
        X=X,
        y=y,
        feature_names=list(X_df.columns),
        target_name=target_col,
        index=idx,
        classes=classes,
    )
