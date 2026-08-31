from __future__ import annotations

from typing import Protocol

import numpy as np


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
