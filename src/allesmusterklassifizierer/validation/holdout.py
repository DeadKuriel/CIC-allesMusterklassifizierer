from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..errors import ValidationError
from .base import Split


@dataclass(frozen=True)
class HoldoutSplitter:
    test_size: float = 0.2
    seed: int = 42
    stratify: bool = True

    def split(self, y: np.ndarray) -> Iterable[Split]:
        n = y.shape[0]
        if n < 2:
            raise ValidationError("Holdout requiere al menos 2 muestras.")

        if not (0.0 < self.test_size < 1.0):
            raise ValidationError("test_size debe estar en (0, 1).")

        rng = np.random.default_rng(self.seed)
        all_idx = np.arange(n, dtype=int)

        test_n = int(round(n * self.test_size))
        test_n = max(1, min(n - 1, test_n))

        if not self.stratify:
            perm = rng.permutation(all_idx)
            test_idx = perm[:test_n]
            train_idx = perm[test_n:]
            yield Split(fold=0, train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))
            return

        # stratified sampling
        unique_classes = np.unique(y)
        test_parts = []
        train_parts = []

        # compute per class test sizes (rounded) and then adjust to match total
        per_class = []
        for c in unique_classes:
            cls_idx = all_idx[y == c]
            if cls_idx.size == 0:
                continue
            per_class.append((c, cls_idx))

        desired = []
        for c, cls_idx in per_class:
            m = cls_idx.size
            k = int(round(m * self.test_size))
            k = max(1 if m > 1 else 0, min(m - 1 if m > 1 else 0, k))
            desired.append([c, cls_idx, k])

        # adjust total
        total = sum(k for _, _, k in desired)
        # If total differs, fix by distributing +/- 1 while respecting bounds
        diff = test_n - total

        # helper bounds
        def min_k(m: int) -> int:
            return 0 if m == 1 else 1

        def max_k(m: int) -> int:
            return 0 if m == 1 else m - 1

        i = 0
        while diff != 0 and desired:
            c, cls_idx, k = desired[i]
            m = cls_idx.size
            if diff > 0:
                if k < max_k(m):
                    desired[i][2] += 1
                    diff -= 1
            else:
                if k > min_k(m):
                    desired[i][2] -= 1
                    diff += 1
            i = (i + 1) % len(desired)

        for c, cls_idx, k in desired:
            perm = rng.permutation(cls_idx)
            test_idx_c = perm[:k]
            train_idx_c = perm[k:]
            test_parts.append(test_idx_c)
            train_parts.append(train_idx_c)

        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)

        if test_idx.size == 0 or train_idx.size == 0:
            raise ValidationError("Holdout estratificado produjo split vacío. Revisa tu dataset/test_size.")

        yield Split(fold=0, train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))
