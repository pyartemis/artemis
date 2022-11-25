from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog, ProblemType
from artemis.utilities.metrics import Metric, RMSE


class PermutationImportance(VariableImportanceMethod):

    def __init__(self, metric: Metric = RMSE()):
        super().__init__(ImportanceMethod.PERMUTATION_IMPORTANCE)
        self.metric = metric

    def importance(
            self,
            model,
            X: pd.DataFrame,
            y_true: np.array = None,
            n_repeat: int = 5,
            features: List[str] = None,
            show_progress: bool = False,
    ):
        self.variable_importance = _permutation_importance(model, X, y_true, self.metric,
                                                           n_repeat, features,
                                                           show_progress)
        return self.variable_importance


def _permutation_importance(model, X: pd.DataFrame, y: np.array, metric: Metric, n_repeat: int, features: List[str],
                            show_progress: bool):
    base_score = metric.calculate(y, model.predict(X))
    corrupted_scores = _corrupted_scores(model, X, y, features, metric, n_repeat, show_progress)

    feature_importance = [
        {"Feature": f, "Value": _neg_if_class(metric, np.mean(corrupted_scores[f]) - base_score)}
        for f in corrupted_scores.keys()
    ]

    return pd.DataFrame.from_records(feature_importance).sort_values(
        by="Value", ascending=False, ignore_index=True
    )


def _corrupted_scores(model, X: pd.DataFrame, y: np.array, features: List[str], metric: Metric, n_repeat: int,
                      show_progress: bool):
    X_copy_permuted = X.copy()
    corrupted_scores = {f: [] for f in features}
    for _ in tqdm(range(n_repeat), disable=not show_progress, desc=ProgressInfoLog.CALC_VAR_IMP):
        for feature in features:
            X_copy_permuted[feature] = np.random.permutation(X_copy_permuted[feature])
            corrupted_scores[feature].append(metric.calculate(y, model.predict(X_copy_permuted)))
            X_copy_permuted[feature] = X[feature]

    return corrupted_scores


def _neg_if_class(metric: Metric, value: float):
    if metric.applicable_to(ProblemType.CLASSIFICATION):
        return -value

    return value
