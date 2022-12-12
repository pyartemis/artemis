from itertools import combinations
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PermutationImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import InteractionMethod, ProblemType, ProgressInfoLog
from artemis.utilities.performance_metrics import Metric, RMSE
from artemis.utilities.ops import all_if_none, sample_both_if_not_none


class SejongOhMethod(FeatureInteractionMethod):
    """
    Sejong Oh's Performance Based Method for Feature Interaction Extraction. 
    
    Attributes:
    ----------
    method : str 
        Method name, used also for naming column with results in `ovo` pd.DataFrame.
    visualizer : Visualizer
        Object providing visualization. Automatically created on the basis of a method and used to create visualizations.
    ovo : pd.DataFrame 
        One versus one (pair) feature interaction values. 
    feature_importance : pd.DataFrame 
        Feature importance values.
    metric : Metric
        Metric used for calculating performance.
    model : object
        Explained model.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    y_sampled: np.array or pd.Series
        Sampled target values used for calculation.
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.
    random_state : int
        Random state used for reproducibility.

    References:
    ----------
    - https://www.mdpi.com/2076-3417/9/23/5191
    """

    def __init__(self, metric: Metric = RMSE(), random_state: Optional[int] = None):
        """Constructor for SejongOhMethod
        
        Parameters:
        ----------
        metric : Metric
            Metric used to calculate model performance. Defaults to RMSE().
        random_state : int, optional 
            Random state for reproducibility. Defaults to None.
        """
        super().__init__(InteractionMethod.PERFORMANCE_BASED, random_state)
        self.metric = metric
        self.y_sampled = None

    @property
    def interactions_ascending_order(self):
        return False

    def fit(
            self,
            model,
            X: pd.DataFrame,
            y_true: Union[np.array, pd.Series],  
            n: int = None,
            n_repeat: int = 10,
            features: List[str] = None,
            show_progress: bool = False,
    ):  
        """Calculates Performance Based Feature Interactions Strength and Permutation Based Feature Importance for the given model.

        Parameters:
        ----------
        model : object
            Model to be explained, should have predict method.
        X : pd.DataFrame
            Data used to calculate interactions. If n is not None, n rows from X will be sampled. 
        y_true : np.array or pd.Series
            Target values for X data. 
        n : int, optional
            Number of samples to be used for calculation of interactions. If None, all rows from X will be used. Default is None.
        n_repeat : int, optional
            Number of permutations. Default is 10.
        features : List[str], optional
            List of features for which interactions will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        """
        self.X_sampled, self.y_sampled = sample_both_if_not_none(self._random_generator, X, y_true, n)
        self.features_included = all_if_none(X.columns, features)
        self.pairs = list(combinations(self.features_included, 2))
        self.ovo = _perf_based_ovo(self, model, self.X_sampled, self.y_sampled, n_repeat, show_progress)

        # calculate variable importance
        self._feature_importance_obj = PermutationImportance(self.metric)
        self.feature_importance = self._feature_importance_obj.importance(model, X=self.X_sampled,
                                                                            y_true=self.y_sampled,
                                                                            n_repeat=n_repeat,
                                                                            features=self.features_included,
                                                                            show_progress=show_progress)


def _perf_based_ovo(
        method_class: SejongOhMethod, model, X: pd.DataFrame, y_true: np.array, n_repeat: int, show_progress: bool
):
    """For each pair of `features_included`, calculate Sejong Oh performance based interaction value."""
    original_performance = method_class.metric.calculate(y_true, model.predict(X))
    interactions = list()

    for f1, f2 in tqdm(method_class.pairs, disable=not show_progress, desc=ProgressInfoLog.CALC_OVO):
        inter = [
            np.abs(_inter(method_class, model, X, y_true, f1, f2, original_performance)) for _ in range(n_repeat)
        ]
        interactions.append([f1, f2, np.mean(inter)])

    return pd.DataFrame(interactions, columns=["Feature 1", "Feature 2", method_class.method]).sort_values(
        by=method_class.method, key=abs, ascending=method_class.interactions_ascending_order, ignore_index=True
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
    with respect to when both are permuted together.

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
    p = method_class._random_generator.permutation(len(X))

    for feature in features:
        X_copy_permuted[feature] = X_copy_permuted[feature].values[p]

    return _neg_if_class(
        method_class,
        method_class.metric.calculate(y_true, model.predict(X_copy_permuted)) - reference_performance,
    )


def _neg_if_class(method_class: SejongOhMethod, value: float):
    """Classification metrics should be maximized."""
    if method_class.metric.applicable_to(ProblemType.CLASSIFICATION):
        return -value

    return value
