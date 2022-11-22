from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod
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
        return _permutation_importance(model, X, y_true, self.metric,
                                       n_repeat, features,
                                       show_progress, self.method)


def _permutation_importance(model, X, y, metric, n_repeat, features, show_progress, method):
    base_score = metric.calculate(y, model.predict(X))
    corrupted_scores = _corrupted_scores(X, y, features, metric, model, n_repeat, show_progress)

    feature_importance = [
        {"Feature": f, method: np.mean(corrupted_scores[f]) - base_score}
        for f in corrupted_scores.keys()
    ]

    return pd.DataFrame.from_records(feature_importance).sort_values(
        by=method, ascending=False, ignore_index=True
    )


def _corrupted_scores(X, y, features, metric, model, n_repeat, show_progress):
    X_copy_permuted = X.copy()
    corrupted_scores = {f: [] for f in features}
    for _ in tqdm(range(n_repeat), disable=not show_progress, desc="Calculating variable importance"):
        for feature in features:
            X_copy_permuted[feature] = np.random.permutation(X_copy_permuted[feature])
            corrupted_scores[feature].append(metric.calculate(y, model.predict(X_copy_permuted)))
            X_copy_permuted[feature] = X[feature]

    return corrupted_scores
