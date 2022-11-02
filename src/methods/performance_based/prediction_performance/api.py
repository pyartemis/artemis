from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.domain.domain import Method, ProblemType
from src.domain.metrics import Metric, RMSE
from src.methods.method import FeatureInteractionMethod
from src.util.ops import all_if_none, sample_both_if_not_none


class SejongOhInteraction(FeatureInteractionMethod):

    def __init__(self, metric: Metric = RMSE()):
        super().__init__(Method.PERFORMANCE_BASED)
        self.metric = metric
        self.y_sampled = None

    def fit(self,
            model,
            X: pd.DataFrame,
            y_true: np.array = None,  # to comply with signature
            n: int = None,
            n_repeat: int = 10,
            features: List[str] = None,
            show_progress: bool = False):
        self.X_sampled, self.y_sampled = sample_both_if_not_none(X, y_true, n)
        self.features_included = all_if_none(X, features)
        self.ovo = self._perf_based_ovo(model, self.X_sampled, self.y_sampled, n_repeat, show_progress)

    def _perf_based_ovo(self, model, X: pd.DataFrame, y_true: np.array, n_repeat: int, show_progress: bool):
        original_performance = self.metric.calculate(y_true, model.predict(X))
        pairs = list(combinations(self.features_included, 2))
        interactions = list()

        for f1, f2 in tqdm(pairs, disable=not show_progress):
            inter = [
                self._inter(model, X, y_true, f1, f2, original_performance)
                for _ in range(n_repeat)
            ]
            interactions.append([f1, f2, sum(inter) / len(inter)])

        return pd.DataFrame(interactions, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=False, ignore_index=True
        )

    def _inter(self, model, X: pd.DataFrame, y_true: np.array, f1: str, f2: str, reference_performance: float):
        score_f1_permuted = self._permute_score(model, X, y_true, [f1], reference_performance)
        score_f2_permuted = self._permute_score(model, X, y_true, [f2], reference_performance)
        score_f1_f2_permuted = self._permute_score(model, X, y_true, [f1, f2], reference_performance)

        return self._neg_if_class(score_f1_f2_permuted - score_f1_permuted - score_f2_permuted)

    def _permute_score(self, model, X: pd.DataFrame, y_true: np.array, features: List[str],
                       reference_performance: float):
        X_copy_permuted = X.copy()

        for feature in features:
            X_copy_permuted[feature] = np.random.permutation(X_copy_permuted[feature])

        return self._neg_if_class(
            self.metric.calculate(y_true, model.predict(X_copy_permuted)) - reference_performance)

    def _neg_if_class(self, value: float):
        if self.metric.applicable_to(ProblemType.CLASSIFICATION):
            return -value

        return value
