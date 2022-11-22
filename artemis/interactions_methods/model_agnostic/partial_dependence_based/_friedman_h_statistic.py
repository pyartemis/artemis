from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.utilities.domain import VisualisationType, InteractionMethod, InteractionCalculationStrategy
from ._pdp import PartialDependenceBasedMethod
from artemis.utilities.ops import remove_element, center, partial_dependence_value


class FriedmanHStatisticMethod(PartialDependenceBasedMethod):

    def __init__(self):
        super().__init__(InteractionMethod.H_STATISTIC)
        self.ova = None
        self._pdp_cache = dict()

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False,
            **kwargs):
        super().fit(model, X, n, features, show_progress, self._pdp_cache)
        self.ova = self._ova(model, self.X_sampled, show_progress, self.features_included)

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        assert self.ovo is not None and self.ova is not None, "Before executing plot() method, fit() must be executed!"

        self.visualisation.plot(self.ovo, vis_type, self.ova)

    def _ova(self, model, X: pd.DataFrame, progress: bool, features: List[str]) -> pd.DataFrame:
        h_stat_one_vs_all = [
            [column, self._calculate_i_versus(model, X, column, remove_element(X.columns, column))]
            for column in tqdm(features, desc=InteractionCalculationStrategy.ONE_VS_ALL, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=False, ignore_index=True
        )

    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        pd_i_list = np.array([])
        pd_versus_list = np.array([])
        pd_i_versus_list = np.array([])

        for _, row in X_sampled.iterrows():
            change_i = {i: row[i]}
            change_versus = {col: row[col] for col in versus}
            change_i_versus = {**change_i, **change_versus}

            pd_i = partial_dependence_value(X_sampled, change_i, model.predict)
            pd_versus = partial_dependence_value(X_sampled, change_versus, model.predict)
            pd_i_versus = partial_dependence_value(X_sampled, change_i_versus, model.predict)

            pd_i_list = np.append(pd_i_list, pd_i)
            pd_versus_list = np.append(pd_versus_list, pd_versus)
            pd_i_versus_list = np.append(pd_i_versus_list, pd_i_versus)

        self._pdp_cache[i] = pd_i_list
        nominator = (center(pd_i_versus_list) - center(pd_i_list) - center(pd_versus_list)) ** 2
        denominator = center(pd_i_versus_list) ** 2
        return np.sum(nominator) / np.sum(denominator)
