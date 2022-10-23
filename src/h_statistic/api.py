from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.domain.domain import ONE_VS_ALL, ONE_VS_ONE, Methods
from src.visualisation import Visualisation
from src.util.ops import remove_element, center, sample_if_not_none
from src.util.partial_dependence import partial_dependence_value


class HStatistic:

    def __init__(self):
        self.ovo = None
        self.ova = None

    def fit(self, model, X: pd.DataFrame, n: int = None, show_progress: bool = False):
        X_sampled = sample_if_not_none(X, n)
        self.ovo, self.ova = self._ovo(model, X_sampled, show_progress), self._ova(model, X_sampled, show_progress)

    def plot(self):
        assert self.ovo is not None and self.ova is not None, "Before executing plot() method, fit() must be executed!"

        Visualisation(method=Methods.H_STATISTIC).plot(self.ova, self.ovo)

    def _ovo(self, model, X: pd.DataFrame, progress: bool) -> pd.DataFrame:
        pairs = list(combinations(X.columns, 2))
        h_stat_pairs = [
            [c1, c2, self._calculate_h_stat_i_versus(model, X, c1, [c2])]
            for c1, c2 in tqdm(pairs, desc=ONE_VS_ONE.value, disable=not progress)
        ]

        return pd.DataFrame(h_stat_pairs, columns=["Feature 1", "Feature 2", Methods.H_STATISTIC]).sort_values(
            by=Methods.H_STATISTIC, ascending=False, ignore_index=True
        )

    def _ova(self, model, X: pd.DataFrame, progress) -> pd.DataFrame:
        h_stat_one_vs_all = [
            [column, self._calculate_h_stat_i_versus(model, X, column, remove_element(X.columns, column))]
            for column in tqdm(X.columns, desc=ONE_VS_ALL.value, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", Methods.H_STATISTIC]).sort_values(
            by=Methods.H_STATISTIC, ascending=False, ignore_index=True
        )

    @staticmethod
    def _calculate_h_stat_i_versus(model, X: pd.DataFrame, i: str, versus: List[str]) -> float:
        pd_i_list = np.array([])
        pd_versus_list = np.array([])
        pd_i_versus_list = np.array([])

        for _, row in X.iterrows():
            change_i = {i: row[i]}
            change_versus = {col: row[col] for col in versus}
            change_i_versus = {**change_i, **change_versus}

            pd_i = partial_dependence_value(X, change_i, model.predict)
            pd_versus = partial_dependence_value(X, change_versus, model.predict)
            pd_i_versus = partial_dependence_value(X, change_i_versus, model.predict)

            pd_i_list = np.append(pd_i_list, pd_i)
            pd_versus_list = np.append(pd_versus_list, pd_versus)
            pd_i_versus_list = np.append(pd_i_versus_list, pd_i_versus)

        nominator = (center(pd_i_versus_list) - center(pd_i_list) - center(pd_versus_list)) ** 2
        denominator = center(pd_i_versus_list) ** 2
        return np.sum(nominator) / np.sum(denominator)
