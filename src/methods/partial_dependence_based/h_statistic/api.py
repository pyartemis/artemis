from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.domain.domain import VisualisationType, Method, InteractionCalculationStrategy
from src.methods.partial_dependence_based.pdp import PartialDependenceBasedMethod
from src.util.ops import remove_element, center


class FriedmanHStatistic(PartialDependenceBasedMethod):

    def __init__(self):
        super().__init__(Method.H_STATISTIC)
        self.ova = None

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False):
        super().sample_ovo(model, X, n, features, show_progress)
        self.ova = self._ova(model, self.X_sampled, show_progress, self.features_included)

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        assert self.ovo is not None and self.ova is not None, "Before executing plot() method, fit() must be executed!"

        self.visualisation.plot(self.ovo, vis_type, self.ova)

    def _ova(self, model, X: pd.DataFrame, progress: bool, features: List[str]) -> pd.DataFrame:
        h_stat_one_vs_all = [
            [column, self._calculate_i_versus(model, X, column, remove_element(X.columns, column))]
            for column in tqdm(features, desc=InteractionCalculationStrategy.ONE_VS_ALL, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", Method.H_STATISTIC]).sort_values(
            by=Method.H_STATISTIC, ascending=False, ignore_index=True
        )

    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        pd_i_list = np.array([])
        pd_versus_list = np.array([])
        pd_i_versus_list = np.array([])

        for _, row in X_sampled.iterrows():
            change_i = {i: row[i]}
            change_versus = {col: row[col] for col in versus}
            change_i_versus = {**change_i, **change_versus}

            pd_i = PartialDependenceBasedMethod.partial_dependence_value(X_sampled, change_i, model.predict)
            pd_versus = PartialDependenceBasedMethod.partial_dependence_value(X_sampled, change_versus, model.predict)
            pd_i_versus = PartialDependenceBasedMethod.partial_dependence_value(X_sampled, change_i_versus,
                                                                                model.predict)

            pd_i_list = np.append(pd_i_list, pd_i)
            pd_versus_list = np.append(pd_versus_list, pd_versus)
            pd_i_versus_list = np.append(pd_i_versus_list, pd_i_versus)

        nominator = (center(pd_i_versus_list) - center(pd_i_list) - center(pd_versus_list)) ** 2
        denominator = center(pd_i_versus_list) ** 2
        return np.sum(nominator) / np.sum(denominator)
