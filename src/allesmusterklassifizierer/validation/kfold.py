from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from ..errors import ValidationError
from .base import Split


@dataclass(frozen=True)
class KFoldSplitter:
    """
    K-Fold with optional stratification.
    - If stratify=True: indices are distributed across folds in a class-balanced manner.
    - shuffle=True + seed: performs shuffling either within each class (when stratified)
    or globally (when not stratified).
    """
    n_splits: int = 5
    seed: int = 42
    shuffle: bool = True
    stratify: bool = True

    def split(self, y: np.ndarray) -> Iterable[Split]:
        n = int(y.shape[0])
        if n < 2:
            raise ValidationError("K-Fold requiere al menos 2 muestras.")
        if not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValidationError("n_splits debe ser entero >= 2.")
        if self.n_splits > n:
            raise ValidationError(f"n_splits={self.n_splits} no puede ser mayor que n_samples={n}.")

        rng = np.random.default_rng(self.seed)
        all_idx = np.arange(n, dtype=int)

        if not self.stratify:
            idx = all_idx.copy()
            if self.shuffle:
                idx = rng.permutation(idx)
            folds = _chunk_round_robin(idx, self.n_splits) # type: ignore
            for fold_id in range(self.n_splits):
                test_idx = np.sort(folds[fold_id])
                train_idx = np.sort(np.concatenate([folds[j] for j in range(self.n_splits) if j != fold_id]))
                if test_idx.size == 0 or train_idx.size == 0:
                    raise ValidationError("K-Fold produjo un fold vacío. Revisa n_splits.")
                yield Split(fold=fold_id, train_idx=train_idx, test_idx=test_idx)
            return

        # Stratified K-Fold
        folds: List[List[int]] = [[] for _ in range(self.n_splits)]
        classes = np.unique(y)

        for c in classes:
            cls_idx = all_idx[y == c]
            if cls_idx.size == 0:
                continue
            if self.shuffle:
                cls_idx = rng.permutation(cls_idx)
            for i, ix in enumerate(cls_idx):
                folds[i % self.n_splits].append(int(ix))

        # Convert to arrays and yield
        fold_arrays = [np.array(sorted(f), dtype=int) for f in folds]
        for fold_id in range(self.n_splits):
            test_idx = fold_arrays[fold_id]
            train_idx = np.sort(np.concatenate([fold_arrays[j] for j in range(self.n_splits) if j != fold_id]))
            if test_idx.size == 0 or train_idx.size == 0:
                raise ValidationError(
                    "K-Fold estratificado produjo un fold vacío. "
                    "Posibles causas: clase con muy pocos ejemplos vs n_splits."
                )
            yield Split(fold=fold_id, train_idx=train_idx, test_idx=np.sort(test_idx))


def _chunk_round_robin(idx: np.ndarray, n_splits: int) -> List[np.ndarray]:
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for i, ix in enumerate(idx):
        folds[i % n_splits].append(int(ix))
    return [np.array(f, dtype=int) for f in folds]
