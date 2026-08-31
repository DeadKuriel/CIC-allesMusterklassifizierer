from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class Splitter(Protocol):
    def split(self, y: np.ndarray) -> Iterable[Split]: ...
