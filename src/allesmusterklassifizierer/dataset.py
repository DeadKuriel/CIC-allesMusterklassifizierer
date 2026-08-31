from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataError, SchemaError
from .v1config import ExperimentConfig


@dataclass
class TabularDataset:
    frame: pd.DataFrame
    original_index: pd.Series
    sha256: str
    source: Path

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.drop(columns=[c for c in self.frame.columns if c not in self.feature_names])

    @property
    def feature_names(self) -> list[str]:
        return list(self.frame.attrs["feature_names"])

    @property
    def y(self) -> pd.Series:
        return self.frame[self.frame.attrs["target"]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    extensions = {".csv": "csv", ".tsv": "tsv", ".parquet": "parquet", ".xlsx": "excel", ".xls": "excel"}
    try:
        return extensions[path.suffix.lower()]
    except KeyError as exc:
        raise DataError(f"Formato no soportado para '{path}'") from exc


def _duplicate_headers(path: Path, cfg: ExperimentConfig, fmt: str) -> list[str]:
    if fmt not in {"csv", "tsv"}:
        return []
    import csv
    delimiter = cfg.dataset.delimiter or ("\t" if fmt == "tsv" else ",")
    with path.open(encoding=cfg.dataset.encoding, newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    return sorted({name for name in header if header.count(name) > 1})


def load_dataset(cfg: ExperimentConfig, *, require_target: bool = True) -> TabularDataset:
    path = cfg.dataset.path
    if not path.is_file():
        raise DataError(f"No existe el dataset: {path}")
    fmt = _format(path, cfg.dataset.format)
    duplicates = _duplicate_headers(path, cfg, fmt)
    if duplicates:
        raise SchemaError(f"Nombres de columna duplicados: {duplicates}")
    na_values = cfg.dataset.missing_values or None
    try:
        if fmt in {"csv", "tsv"}:
            frame = pd.read_csv(path, sep=cfg.dataset.delimiter or ("\t" if fmt == "tsv" else ","),
                                encoding=cfg.dataset.encoding, decimal=cfg.dataset.decimal, na_values=na_values)
        elif fmt == "parquet":
            frame = pd.read_parquet(path)
        elif fmt == "excel":
            frame = pd.read_excel(path, sheet_name=cfg.dataset.sheet_name, na_values=na_values)
        else:  # pragma: no cover
            raise DataError(f"Formato no soportado: {fmt}")
    except ImportError as exc:
        raise DataError(f"El formato {fmt} requiere el extra 'formats'") from exc
    except Exception as exc:
        raise DataError(f"No se pudo leer {path}: {exc}") from exc
    if frame.empty:
        raise DataError("El dataset está vacío")
    if cfg.dataset.row_group_column:
        group_column = cfg.dataset.row_group_column
        if group_column in frame.columns:
            raise SchemaError(f"La columna de grupo derivada ya existe: {group_column}")
        frame[group_column] = np.arange(len(frame)) // int(cfg.dataset.row_group_size)
    declared = [c.name for c in cfg.columns if c.role != "ignored"]
    required = [name for name in declared if require_target or name != cfg.target]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise SchemaError(f"Columnas requeridas ausentes: {missing}")
    feature_names = [c.name for c in cfg.columns if c.role == "feature"]
    empty = [name for name in feature_names if frame[name].isna().all()]
    if empty:
        raise SchemaError(f"Columnas feature completamente vacías: {empty}")
    for col in cfg.columns:
        if col.name not in frame or col.role in {"ignored", "target"}:
            continue
        if col.semantic_type == "numeric":
            converted = pd.to_numeric(frame[col.name], errors="coerce")
            introduced = converted.isna() & frame[col.name].notna()
            if introduced.any():
                raise SchemaError(f"Valores incompatibles con numeric en '{col.name}'")
            frame[col.name] = converted
            if np.isinf(converted.dropna().to_numpy()).any():
                raise SchemaError(f"Valores infinitos en '{col.name}'")
        elif col.semantic_type == "datetime":
            converted = pd.to_datetime(frame[col.name], errors="coerce")
            if (converted.isna() & frame[col.name].notna()).any():
                raise SchemaError(f"Fechas inválidas en '{col.name}'")
            frame[col.name] = converted
        elif col.semantic_type == "ordinal":
            unknown = set(frame[col.name].dropna().unique()) - set(col.order or [])
            if unknown:
                raise SchemaError(f"Categorías ordinales desconocidas en '{col.name}': {sorted(map(str, unknown))}")
    if require_target:
        if frame[cfg.target].isna().any() and cfg.audit.target_missing == "fail":
            raise SchemaError("El objetivo contiene valores faltantes; target_missing=fail")
        if frame[cfg.target].nunique(dropna=True) < 2:
            raise SchemaError("El objetivo debe contener al menos dos clases")
    original_index = pd.Series(frame.index, index=frame.index, name="original_index")
    frame.attrs["feature_names"] = feature_names
    frame.attrs["target"] = cfg.target
    return TabularDataset(frame=frame, original_index=original_index, sha256=file_sha256(path), source=path)


def validate_prediction_schema(frame: pd.DataFrame, required: list[str], *, extra: str = "ignore") -> pd.DataFrame:
    missing = [c for c in required if c not in frame]
    if missing:
        raise SchemaError(f"Columnas requeridas ausentes: {missing}")
    extras = [c for c in frame if c not in required]
    if extras and extra == "fail":
        raise SchemaError(f"Columnas adicionales no permitidas: {extras}")
    return frame.loc[:, required]
