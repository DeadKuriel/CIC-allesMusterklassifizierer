from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass
