from itertools import combinations
from statistics import stdev
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.domain.domain import ONE_VS_ONE, Methods
from src.visualisation import Visualisation
from src.util.ops import sample_if_not_none
from src.util.partial_dependence import partial_dependence_value


class VariableInteraction:
    def __init__(self):
        self.ovo = None

    def fit(self, model, X: pd.DataFrame, n: int = None, show_progress: bool = False):
        X_sampled = sample_if_not_none(X, n)
        self.ovo = self._ovo(model, X_sampled, show_progress)

    def plot(self):
        assert self.ovo is not None, "Before executing plot() method, fit() must be executed!"

        Visualisation(Methods.VARIABLE_INTERACTION).plot(self.ovo)

    def _ovo(self, model, X: pd.DataFrame, progress: bool) -> pd.DataFrame:
        pairs = list(combinations(X.columns, 2))
        vint_pairs = [
            [c1, c2, self._calculate_vint_ij(model, X, c1, c2)]
            for c1, c2 in tqdm(pairs, desc=ONE_VS_ONE.value, disable=not progress)
        ]

        return pd.DataFrame(
            vint_pairs, columns=["Feature 1", "Feature 2", Methods.VARIABLE_INTERACTION]
        ).sort_values(by=Methods.VARIABLE_INTERACTION, ascending=False, ignore_index=True)

    @staticmethod
    def _calculate_vint_ij(model, X: pd.DataFrame, i: str, j: str) -> float:
        pd_values = np.array(
            [
                [partial_dependence_value(X, {i: x_i, j: x_j}, model.predict) for x_i in set(X[i])]
                for x_j in set(X[j])
            ]
        )
        res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
        res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
        return (res_j + res_i) / 2
