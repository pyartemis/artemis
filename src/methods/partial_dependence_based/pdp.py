from abc import abstractmethod
from itertools import combinations
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from src.domain.domain import InteractionCalculationStrategy
from src.methods.method import FeatureInteractionMethod
from src.util.ops import sample_if_not_none, all_if_none


class PartialDependenceBasedMethod(FeatureInteractionMethod):

    def __init__(self, method: str):
        super().__init__(method)

    def sample_ovo(self,
                   model,
                   X: pd.DataFrame,
                   n: int = None,
                   features: List[str] = None,
                   show_progress: bool = False):
        self.X_sampled = sample_if_not_none(X, n)
        self.features_included = all_if_none(X, features)

        self.ovo = self._ovo(model, self.X_sampled, show_progress, self.features_included)

    def _ovo(self, model, X_sampled: pd.DataFrame, show_progress: bool, features: List[str]):
        pairs = list(combinations(features, 2))
        value_pairs = [
            [c1, c2, self._calculate_i_versus(model, X_sampled, c1, [c2])]
            for c1, c2 in tqdm(pairs, desc=InteractionCalculationStrategy.ONE_VS_ONE, disable=not show_progress)
        ]

        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=False, ignore_index=True
        )

    @staticmethod
    def partial_dependence_value(df: pd.DataFrame, change_dict: Dict, predict_function: Callable) -> ndarray:
        assert all(column in df.columns for column in change_dict.keys())
        df_changed = df.assign(**change_dict)
        return np.mean(predict_function(df_changed))

    @abstractmethod
    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        ...
