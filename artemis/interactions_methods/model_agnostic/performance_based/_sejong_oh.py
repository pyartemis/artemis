from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PermutationImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import InteractionMethod, ProblemType, ProgressInfoLog
from artemis.utilities.performance_metrics import Metric, RMSE
from artemis.utilities.ops import all_if_none, sample_both_if_not_none


class SejongOhMethod(FeatureInteractionMethod):
    """Class implementing Sejong Oh Performance Based feature interaction method.
    Attributes:
        method (str) -- name of interaction method
        visualizer (Visualizer) -- automatically created on the basis of a method and used to create visualizations
        variable_importance (pd.DataFrame) -- variable importance values 
        ovo (pd.DataFrame) -- one versus one variable interaction values 
        metric (Metric) -- metric used to calculate interactions
    
    References:
    - https://www.mdpi.com/2076-3417/9/23/5191
    """

    def __init__(self, metric: Metric = RMSE()):
        super().__init__(InteractionMethod.PERFORMANCE_BASED)
        self.metric = metric
        self.y_sampled = None
    
    @property
    def interactions_ascending_order(self):
        return False

    def fit(
            self,
            model,
            X: pd.DataFrame,
            y_true: np.array = None,  # to comply with signature
            n: int = None,
            n_repeat: int = 10,
            features: List[str] = None,
            show_progress: bool = False,
    ):

        """Calculates Performance Based Interactions and Permutationl Based Feature Importance for the given model. 

        Parameters:
            model -- model to be explained
            X (pd.DataFrame, optional) -- data used to calculate interactions
            y_true (np.array) -- target values for X data
            n (int, optional) -- number of samples to be used for calculation of interactions
            n_repeat (int) -- number of permutations 
            features (List[str], optional) -- list of features for which interactions will be calculated
            show_progress (bool) -- whether to show progress bar 
        """
        self.X_sampled, self.y_sampled = sample_both_if_not_none(X, y_true, n)
        self.features_included = all_if_none(X, features)
        self.ovo = _perf_based_ovo(self, model, self.X_sampled, self.y_sampled, n_repeat, show_progress)

        # calculate variable importance
        permutation_importance = PermutationImportance(self.metric)
        self.variable_importance = permutation_importance.importance(model, X=self.X_sampled, y_true=self.y_sampled,
                                                                     n_repeat=n_repeat,
                                                                     features=self.features_included,
                                                                     show_progress=show_progress)


def _perf_based_ovo(
        method_class: SejongOhMethod, model, X: pd.DataFrame, y_true: np.array, n_repeat: int, show_progress: bool
):
    """For each pair of `features_included`, calculate Sejong Oh performance based interaction value."""
    original_performance = method_class.metric.calculate(y_true, model.predict(X))
    pairs = list(combinations(method_class.features_included, 2))
    interactions = list()

    for f1, f2 in tqdm(pairs, disable=not show_progress, desc=ProgressInfoLog.CALC_OVO):
        inter = [
            _inter(method_class, model, X, y_true, f1, f2, original_performance) for _ in range(n_repeat)
        ]
        interactions.append([f1, f2, abs(sum(inter) / len(inter))])

    return pd.DataFrame(interactions, columns=["Feature 1", "Feature 2", method_class.method]).sort_values(
        by=method_class.method, ascending=method_class.interactions_ascending_order, ignore_index=True
    )


def _inter(
        method_class: SejongOhMethod,
        model,
        X: pd.DataFrame,
        y_true: np.array,
        f1: str,
        f2: str,
        reference_performance: float,
):
    """
    Calculates performance-based interaction between features `f1` and `f2`.
    Intuitively, it calculates the impact on the performance of the model, when one of [f1, f2] are permuted
    with respect to when both are permuted.

    Specifics can be found in: https://www.mdpi.com/2076-3417/9/23/5191.
    """
    score_f1_permuted = _permute_score(method_class, model, X, y_true, [f1], reference_performance)
    score_f2_permuted = _permute_score(method_class, model, X, y_true, [f2], reference_performance)
    score_f1_f2_permuted = _permute_score(method_class, model, X, y_true, [f1, f2], reference_performance)

    return _neg_if_class(method_class, score_f1_f2_permuted - score_f1_permuted - score_f2_permuted)


def _permute_score(
        method_class: SejongOhMethod,
        model,
        X: pd.DataFrame,
        y_true: np.array,
        features: List[str],
        reference_performance: float,
):
    """Permute `features` list and assess performance of the model."""
    X_copy_permuted = X.copy()

    for feature in features:
        X_copy_permuted[feature] = np.random.permutation(X_copy_permuted[feature])

    return _neg_if_class(
        method_class,
        method_class.metric.calculate(y_true, model.predict(X_copy_permuted)) - reference_performance,
    )


def _neg_if_class(method_class: SejongOhMethod, value: float):
    """Classification metrics should be maximized."""
    if method_class.metric.applicable_to(ProblemType.CLASSIFICATION):
        return -value

    return value
