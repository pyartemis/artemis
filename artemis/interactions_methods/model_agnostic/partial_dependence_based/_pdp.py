from abc import abstractmethod
from itertools import combinations
from typing import List

import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import get_predict_function, sample_if_not_none, all_if_none


class PartialDependenceBasedMethod(FeatureInteractionMethod):

    def __init__(self, method: str):
        super().__init__(method)

    @property
    def interactions_ascending_order(self):
        return False

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False,
            pdp_cache: dict = None):
        """
        Calculate one versus one feature interaction and partial dependence based variable importance.

        Args:
            model: model for which interactions will be extracted, must have implemented predict method
            X: data used to calculate interactions
            n: number of rows to be sampled, if None full data will be taken
            features: list of features included in interactions calculation; if None all features will be used
            show_progress: determine whether to show the progress bar
            pdp_cache: previously calculated partial dependence values that can be used to calculate interactions values

        Returns:
            object: None
        """
        self.predict_function = get_predict_function(model)
        self.model = model
        self.sample_ovo(self.predict_function, self.model, X, n, features, show_progress)

        self.variable_importance = PartialDependenceBasedImportance().importance(self.model, self.X_sampled,
                                                                                 features=self.features_included,
                                                                                 show_progress=show_progress,
                                                                                 precalculated_pdp=pdp_cache)

    def sample_ovo(self,
                   predict_function,
                   model,
                   X: pd.DataFrame,
                   n: int = None,
                   features: List[str] = None,
                   show_progress: bool = False):
        self.X_sampled = sample_if_not_none(X, n)
        self.features_included = all_if_none(X, features)

        self.ovo = self._ovo(predict_function, model, self.X_sampled, show_progress, self.features_included)

    def _ovo(self, predict_function, model, X_sampled: pd.DataFrame, show_progress: bool, features: List[str]):
        pairs = list(combinations(features, 2))
        value_pairs = [
            [c1, c2, self._calculate_i_versus(predict_function, model, X_sampled, c1, [c2])]
            for c1, c2 in tqdm(pairs, desc=ProgressInfoLog.CALC_OVO, disable=not show_progress)
        ]

        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=self.interactions_ascending_order, ignore_index=True
        ).fillna(0)

    @abstractmethod
    def _calculate_i_versus(self, predict_function, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        """
        Abstract interaction value calculation between feature (i) and a list of features (versus).
        Derived classes need to implement this method to provide its interaction values.

        Args:
            model: model for which interactions will be extracted, must have implemented predict method
            X_sampled: data used to calculate interactions
            i: distinguished feature for which interactions with versus will be calculated
            versus: list of features for which interactions with feature `i` will be calculated

        Returns:
            value of the interaction
        """
        ...
