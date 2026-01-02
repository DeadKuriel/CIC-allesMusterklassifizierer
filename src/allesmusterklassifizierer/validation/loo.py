from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..errors import ValidationError
from .base import Split


@dataclass(frozen=True)
class LeaveOneOutSplitter:
    def split(self, y: np.ndarray) -> Iterable[Split]:
        n = y.shape[0]
        if n < 2:
            raise ValidationError("LOO requiere al menos 2 muestras.")

        all_idx = np.arange(n, dtype=int)
        for i in range(n):
            test_idx = np.array([i], dtype=int)
            train_idx = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
            yield Split(fold=i, train_idx=train_idx, test_idx=test_idx)
