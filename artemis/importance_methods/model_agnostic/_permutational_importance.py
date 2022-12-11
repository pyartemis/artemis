from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import FeatueImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog, ProblemType
from artemis.utilities.performance_metrics import Metric, RMSE


class PermutationImportance(FeatueImportanceMethod):
    """
    Permutation-Based Feature Importance.
    It is used for calculating feature importance for performance based feature interaction - Sejong Oh method.

    Importance of a feature is defined by the metric selected by user (default is sum of gains).

    Attributes:
    ----------
    method : str 
        Method name.
    metric: Metric
        Metric used for calculating performance.
    feature_importance : pd.DataFrame 
        Feature importance values.
        
    References:
    ----------
    - https://jmlr.org/papers/v20/18-760.html
    """

    def __init__(self, metric: Metric = RMSE(), random_state: Optional[int] = None):
        """Constructor for PermutationImportance.

        Parameters:
        ----------
        metric : Metric
            Metric used to calculate model performance. Defaults to RMSE().
        random_state : int, optional 
            Random state for reproducibility. Defaults to None.
        """
        super().__init__(ImportanceMethod.PERMUTATION_IMPORTANCE, random_state=random_state)
        self.metric = metric

    def importance(
        self,
        model,
        X: pd.DataFrame,
        y_true: np.array,
        n_repeat: int = 15,
        features: Optional[List[str]] = None,
        show_progress: bool = False,
    ):
        """Calculates Permutation Based Feature Importance.

        Parameters:
        ----------
        model : object
               Model for which importance will be calculated, should have predict method.
        X : pd.DataFrame
            Data used to calculate importance. 
        y_true : np.array or pd.Series
            Target values for X data. 
        n_repeat : int, optional
            Number of permutations. Default is 10.
        features : List[str], optional
            List of features for which importance will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.

        Returns:
        -------
        pd.DataFrame
            Result dataframe containing feature importance with columns: "Feature", "Importance"
        """
        self.feature_importance = _permutation_importance(
            model, X, y_true, self.metric, n_repeat, features, show_progress, self._random_generator
        )
        return self.feature_importance
    @property
    def importance_ascending_order(self):
        return False


def _permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.array,
    metric: Metric,
    n_repeat: int,
    features: List[str],
    show_progress: bool,
    random_generator: np.random._generator.Generator
):
    base_score = metric.calculate(y, model.predict(X))
    corrupted_scores = _corrupted_scores(
        model, X, y, features, metric, n_repeat, show_progress, random_generator
    )

    feature_importance = [
        {
            "Feature": f,
            "Importance": _neg_if_class(metric, np.mean(corrupted_scores[f]) - base_score),
        }
        for f in corrupted_scores.keys()
    ]

    return pd.DataFrame.from_records(feature_importance).sort_values(
        by="Importance", ascending=False, ignore_index=True
    )


def _corrupted_scores(
    model,
    X: pd.DataFrame,
    y: np.array,
    features: List[str],
    metric: Metric,
    n_repeat: int,
    show_progress: bool,
    random_generator: np.random._generator.Generator
):
    X_copy_permuted = X.copy()
    corrupted_scores = {f: [] for f in features}
    for _ in tqdm(
        range(n_repeat), disable=not show_progress, desc=ProgressInfoLog.CALC_VAR_IMP
    ):
        for feature in features:
            X_copy_permuted[feature] = random_generator.permutation(X_copy_permuted[feature])
            corrupted_scores[feature].append(
                metric.calculate(y, model.predict(X_copy_permuted))
            )
            X_copy_permuted[feature] = X[feature]

    return corrupted_scores


def _neg_if_class(metric: Metric, value: float):
    if metric.applicable_to(ProblemType.CLASSIFICATION):
        return -value

    return value
