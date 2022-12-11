from abc import abstractmethod
from typing import Optional

import pandas as pd
import numpy as np


class FeatureImportanceMethod:
    """
    Abstract base class for Feature Importance methods. 
    This class should not be used directly. Use derived classes instead.
    
    Attributes:
    ----------
        method : str 
            Method name.
        feature_importance : pd.DataFrame 
            Feature importance values.
    """

    def __init__(self, method: str, random_state: Optional[int] = None):
        self.method = method
        self.feature_importance = None
        self._random_generator = np.random.default_rng(random_state)

    @property
    @abstractmethod
    def importance_ascending_order(self) -> bool:
        ...

    @abstractmethod
    def importance(self, model, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
