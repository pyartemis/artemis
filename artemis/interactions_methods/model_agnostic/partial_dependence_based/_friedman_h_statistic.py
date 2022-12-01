from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.utilities.domain import VisualisationType, InteractionMethod, ProgressInfoLog
from artemis.utilities.ops import remove_element, center, partial_dependence_value
from ._pdp import PartialDependenceBasedMethod


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
        self.ova = self._ova(self.predict_function, self.model, self.X_sampled, show_progress, self.features_included)

    def plot(self, vis_type: str = VisualisationType.HEATMAP, figsize: tuple = (8, 6), show: bool = True):
        assert self.ovo is not None and self.ova is not None, "Before executing plot() method, fit() must be executed!"

        self.visualisation.plot(self.ovo, vis_type, self.ova, self.variable_importance, figsize=figsize, show=show)

    def _ova(self, predict_function, model, X: pd.DataFrame, progress: bool, features: List[str]) -> pd.DataFrame:
        h_stat_one_vs_all = [
            [column, self._calculate_i_versus(predict_function, model, X, column, remove_element(X.columns, column))]
            for column in tqdm(features, desc=ProgressInfoLog.CALC_OVA, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=False, ignore_index=True
        )

    def _calculate_i_versus(self, predict_function, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        pd_i_list = np.array([])
        pd_versus_list = np.array([])
        pd_i_versus_list = np.array([])

        for _, row in X_sampled.iterrows():
            change_i = {i: row[i]}
            change_versus = {col: row[col] for col in versus}
            change_i_versus = {**change_i, **change_versus}

            key_i = _pdp_cache_key(i, row)
            pd_i = _take_from_cache_or_calc(self._pdp_cache, key_i, X_sampled, change_i, predict_function, model)
            self._pdp_cache[key_i] = pd_i

            if len(versus) == 1:
                key_versus = _pdp_cache_key(versus[0], row)
                pd_versus = _take_from_cache_or_calc(self._pdp_cache, key_versus, X_sampled, change_versus, predict_function, model)
                self._pdp_cache[key_versus] = pd_versus
            else:
                pd_versus = partial_dependence_value(X_sampled, change_versus, predict_function, model)

            pd_i_versus = partial_dependence_value(X_sampled, change_i_versus, predict_function, model)

            pd_i_list = np.append(pd_i_list, pd_i)
            pd_versus_list = np.append(pd_versus_list, pd_versus)
            pd_i_versus_list = np.append(pd_i_versus_list, pd_i_versus)

        nominator = (center(pd_i_versus_list) - center(pd_i_list) - center(pd_versus_list)) ** 2
        denominator = center(pd_i_versus_list) ** 2
        return np.sum(nominator) / np.sum(denominator)


def _pdp_cache_key(column, row):
    return column, row[column]


def _take_from_cache_or_calc(pdp_cache, key, X_sampled, change_dict, predict_function, model):
    return pdp_cache[key] if key in pdp_cache else partial_dependence_value(X_sampled, change_dict, predict_function, model)
