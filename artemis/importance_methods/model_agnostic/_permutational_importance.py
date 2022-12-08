from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog, ProblemType
from artemis.utilities.performance_metrics import Metric, RMSE


class PermutationImportance(VariableImportanceMethod):
    """Class implementing Permutation-Based Feature Importance.
    It is used for calculating feature importance for performance based feature interaction - Sejong Oh method.

    Importance of a feature is defined by the metric selected by user (default is sum of gains).

    References:
    - https://jmlr.org/papers/v20/18-760.html
    """

    def __init__(self, metric: Metric = RMSE()):
        """Constructor for PermutationImportance
        Arguments:
            metric  (Metric) -- performance measure to use when assessing model performance,  one of [RMSE, MSE, Accuracy]
        """
        super().__init__(ImportanceMethod.PERMUTATION_IMPORTANCE)
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
        """Calculate Permutation-Based Feature Importance.

        Arguments:
            model -- model for which importance will be extracted
            X (pd.DataFrame) -- data used to calculate importance
            y_true (np.array) -- target values for `X`
            n_repeat (int) -- amount of permutations to generate
            features (List[str], optional) -- list of features that will be used during importance calculation
            show_progress (bool) -- determine whether to show the progress bar

        Returns:
            pd.DataFrame -- DataFrame containing feature importance with columns: "Feature", "Importance"
        """
        self.variable_importance = _permutation_importance(
            model, X, y_true, self.metric, n_repeat, features, show_progress
        )
        return self.variable_importance
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
):
    base_score = metric.calculate(y, model.predict(X))
    corrupted_scores = _corrupted_scores(
        model, X, y, features, metric, n_repeat, show_progress
    )

    feature_importance = [
        {
            "Feature": f,
            "Value": _neg_if_class(metric, np.mean(corrupted_scores[f]) - base_score),
        }
        for f in corrupted_scores.keys()
    ]

    return pd.DataFrame.from_records(feature_importance).sort_values(
        by="Value", ascending=False, ignore_index=True
    )


def _corrupted_scores(
    model,
    X: pd.DataFrame,
    y: np.array,
    features: List[str],
    metric: Metric,
    n_repeat: int,
    show_progress: bool,
):
    X_copy_permuted = X.copy()
    corrupted_scores = {f: [] for f in features}
    for _ in tqdm(
        range(n_repeat), disable=not show_progress, desc=ProgressInfoLog.CALC_VAR_IMP
    ):
        for feature in features:
            X_copy_permuted[feature] = np.random.permutation(X_copy_permuted[feature])
            corrupted_scores[feature].append(
                metric.calculate(y, model.predict(X_copy_permuted))
            )
            X_copy_permuted[feature] = X[feature]

    return corrupted_scores


def _neg_if_class(metric: Metric, value: float):
    if metric.applicable_to(ProblemType.CLASSIFICATION):
        return -value

    return value
