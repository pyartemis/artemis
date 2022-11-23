from abc import abstractmethod
from itertools import combinations
from typing import List

import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import sample_if_not_none, all_if_none


class PartialDependenceBasedMethod(FeatureInteractionMethod):

    def __init__(self, method: str):
        super().__init__(method, )

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False,
            pdp_cache: dict = None):
        self.sample_ovo(model, X, n, features, show_progress)

        self.variable_importance = PartialDependenceBasedImportance().importance(model, self.X_sampled,
                                                                                 features=self.features_included,
                                                                                 show_progress=show_progress,
                                                                                 precalculated_pdp=pdp_cache)

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
            for c1, c2 in tqdm(pairs, desc=ProgressInfoLog.CALC_OVO, disable=not show_progress)
        ]

        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=False, ignore_index=True
        )

    @abstractmethod
    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        ...
