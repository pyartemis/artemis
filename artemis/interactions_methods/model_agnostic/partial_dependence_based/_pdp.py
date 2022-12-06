from abc import abstractmethod
from itertools import combinations
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import get_predict_function, sample_if_not_none, all_if_none


class PartialDependenceBasedMethod(FeatureInteractionMethod):

    def __init__(self, method: str, random_state: Optional[int] = None):
        super().__init__(method, random_state=random_state)

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
        """Calculates Partial Dependence Based Interactions and Importance for the given model. 

        Parameters:
            model -- model to be explained
            X (pd.DataFrame, optional) -- data used to calculate interactions
            n (int, optional) -- number of samples to be used for calculation of interactions
            features (List[str], optional) -- list of features for which interactions will be calculated
            show_progress (bool) -- whether to show progress bar 
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
        self.X_sampled = sample_if_not_none(self.random_generator, X, n)
        self.features_included = all_if_none(X.columns, features)

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

        Parameters:
            model: model for which interactions will be extracted, must have implemented predict method
            X_sampled: data used to calculate interactions
            i: distinguished feature for which interactions with versus will be calculated
            versus: list of features for which interactions with feature `i` will be calculated

        Returns:
            value of the interaction
        """
        ...
