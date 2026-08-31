from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from .errors import ValidationError
from .v1config import SplitConfig


@dataclass(frozen=True)
class DataSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    validation_idx: np.ndarray | None = None


def _check(sp: DataSplit, groups=None, timestamps=None) -> DataSplit:
    if np.intersect1d(sp.train_idx, sp.test_idx).size:
        raise ValidationError("Intersección entre train y test")
    if sp.validation_idx is not None and (
        np.intersect1d(sp.validation_idx, sp.test_idx).size
        or np.intersect1d(sp.validation_idx, sp.train_idx).size
    ):
        raise ValidationError("Intersección de validation con otra partición")
    if groups is not None and set(np.asarray(groups)[sp.train_idx]) & set(
        np.asarray(groups)[sp.test_idx]
    ):
        raise ValidationError("Grupos compartidos entre train y test")
    if timestamps is not None and len(sp.train_idx) and len(sp.test_idx):
        if np.max(np.asarray(timestamps)[sp.train_idx]) >= np.min(
            np.asarray(timestamps)[sp.test_idx]
        ):
            raise ValidationError(
                "La partición temporal usa futuro para predecir pasado"
            )
    return sp


def make_splits(
    config: SplitConfig, y, *, groups=None, timestamps=None, seed=42
) -> list[DataSplit]:
    y = np.asarray(y)
    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        raise ValidationError("Se requieren al menos dos muestras y dos clases")
    idx = np.arange(n)
    s = config.strategy
    result: list[DataSplit] = []
    try:
        if s in {"holdout", "stratified_holdout", "train_validation_test"}:
            strat = y if s != "holdout" else None
            tr, te = train_test_split(
                idx, test_size=config.test_size, random_state=seed, stratify=strat
            )
            if s == "train_validation_test":
                relative = config.validation_size / (1.0 - config.test_size)
                tr, va = train_test_split(
                    tr, test_size=relative, random_state=seed, stratify=y[tr]
                )
                result = [DataSplit(0, np.sort(tr), np.sort(te), np.sort(va))]
            else:
                result = [DataSplit(0, np.sort(tr), np.sort(te))]
        elif s in {
            "kfold",
            "stratified_kfold",
            "loo",
            "group_kfold",
            "stratified_group_kfold",
        }:
            splitter = {
                "kfold": KFold(
                    config.n_splits,
                    shuffle=config.shuffle,
                    random_state=seed if config.shuffle else None,
                ),
                "stratified_kfold": StratifiedKFold(
                    config.n_splits,
                    shuffle=config.shuffle,
                    random_state=seed if config.shuffle else None,
                ),
                "loo": LeaveOneOut(),
                "group_kfold": GroupKFold(config.n_splits),
                "stratified_group_kfold": StratifiedGroupKFold(
                    config.n_splits, shuffle=config.shuffle, random_state=seed
                ),
            }[s]
            if s == "loo":
                iterator = splitter.split(idx)
            elif "group" in s:
                if groups is None:
                    raise ValidationError("Se requieren groups")
                iterator = splitter.split(idx, y, groups)
            else:
                iterator = splitter.split(idx, y)
            result = [
                DataSplit(i, np.asarray(a), np.asarray(b))
                for i, (a, b) in enumerate(iterator)
            ]
        elif s == "group_holdout":
            if groups is None:
                raise ValidationError("Se requieren groups")
            a, b = next(
                GroupShuffleSplit(
                    1, test_size=config.test_size, random_state=seed
                ).split(idx, y, groups)
            )
            result = [DataSplit(0, np.asarray(a), np.asarray(b))]
        elif s in {"temporal", "walk_forward"}:
            if timestamps is None:
                raise ValidationError("Se requieren timestamps")
            order = np.argsort(np.asarray(timestamps), kind="stable")
            if s == "temporal":
                cut = max(1, min(n - 1, int(n * (1 - config.test_size))))
                result = [DataSplit(0, order[:cut], order[cut:])]
            else:
                chunks = np.array_split(order, config.n_splits + 1)
                result = [
                    DataSplit(i, np.concatenate(chunks[: i + 1]), chunks[i + 1])
                    for i in range(config.n_splits)
                ]
    except ValueError as exc:
        raise ValidationError(f"Partición no factible: {exc}") from exc
    return [
        _check(
            sp,
            groups if "group" in s else None,
            timestamps if s in {"temporal", "walk_forward"} else None,
        )
        for sp in result
    ]
