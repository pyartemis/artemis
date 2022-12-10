from abc import abstractmethod
from typing import Optional

import pandas as pd
import numpy as np


class VariableImportanceMethod:

    def __init__(self, method: str, random_state: Optional[int] = None):
        self.method = method
        self.variable_importance = None
        self.random_generator = np.random.default_rng(random_state)

    @property
    @abstractmethod
    def importance_ascending_order(self) -> bool:
        ...

    @abstractmethod
    def importance(self, model, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
