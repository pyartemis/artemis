from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.utilities.domain import VisualisationType, InteractionMethod, ProgressInfoLog
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.utilities.ops import remove_element, center, partial_dependence_value
from ._pdp import PartialDependenceBasedMethod


class FriedmanHStatisticMethod(PartialDependenceBasedMethod):
    """Class implementing Friedman H-statistic feature interaction method.
    Method is described in the following paper: https://arxiv.org/pdf/0811.1679.pdf.

    Attributes:
        ova         [pd.DataFrame], object used for storing one vs all feature interaction profiles
        _pdp_cache  [Dict], object used for caching partial dependence values calculations

    """

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
        """
        See `fit` documentation in `PartialDependenceBasedMethod`.
        Additionally, it calculates one vs all feature interaction profile.
        """
        super().fit(model, X, n, features, show_progress, self._pdp_cache)
        self.ova = self._ova(model, self.X_sampled, show_progress, self.features_included)

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        """
        See `plot` documentation in `PartialDependenceBasedMethod`.
        Additionally, it passes one vs all feature interaction profile to the visualiser class, to be included
        in visualisations.
        """
        if self.ova is None:
            
            raise MethodNotFittedException(self.method)

        self.visualisation.plot(self.ovo, vis_type, self.ova, variable_importance=self.variable_importance)

    def _ova(self, model, X: pd.DataFrame, progress: bool, features: List[str]) -> pd.DataFrame:
        """
        Calculate interaction values between distinguished feature and all other features.
        Args:
            model:      model for which interactions will be extracted, must have implemented predict method
            X:          data used to calculate interactions
            progress:   determine whether to show the progress bar
            features:   list of features for which one versus all interaction will be calculated

        Returns:
            object: features and their corresponding OVA (One Vs All) feature interaction values

        """
        h_stat_one_vs_all = [
            [column, self._calculate_i_versus(model, X, column, remove_element(X.columns, column))]
            for column in tqdm(features, desc=ProgressInfoLog.CALC_OVA, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=False, ignore_index=True
        )

    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        """Friedmann H-statistic feature interaction specifics can be found in https://arxiv.org/pdf/0811.1679.pdf"""
        pd_i_list = np.array([])
        pd_versus_list = np.array([])
        pd_i_versus_list = np.array([])

        for _, row in X_sampled.iterrows():
            change_i = {i: row[i]}
            change_versus = {col: row[col] for col in versus}
            change_i_versus = {**change_i, **change_versus}

            key_i = _pdp_cache_key(i, row)
            pd_i = _take_from_cache_or_calc(self._pdp_cache, key_i, X_sampled, change_i, model)
            self._pdp_cache[key_i] = pd_i

            if len(versus) == 1:
                key_versus = _pdp_cache_key(versus[0], row)
                pd_versus = _take_from_cache_or_calc(self._pdp_cache, key_versus, X_sampled, change_versus, model)
                self._pdp_cache[key_versus] = pd_versus
            else:
                pd_versus = partial_dependence_value(X_sampled, change_versus, model.predict)

            pd_i_versus = partial_dependence_value(X_sampled, change_i_versus, model.predict)

            pd_i_list = np.append(pd_i_list, pd_i)
            pd_versus_list = np.append(pd_versus_list, pd_versus)
            pd_i_versus_list = np.append(pd_i_versus_list, pd_i_versus)

        nominator = (center(pd_i_versus_list) - center(pd_i_list) - center(pd_versus_list)) ** 2
        denominator = center(pd_i_versus_list) ** 2
        return np.sum(nominator) / np.sum(denominator)


def _pdp_cache_key(column, row):
    return column, row[column]


def _take_from_cache_or_calc(pdp_cache, key, X_sampled, change_dict, model):
    return pdp_cache[key] if key in pdp_cache else partial_dependence_value(X_sampled, change_dict, model.predict)
