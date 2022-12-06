from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.utilities.domain import VisualizationType, InteractionMethod, ProgressInfoLog
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.utilities.ops import remove_element, center, partial_dependence_value
from ._pdp import PartialDependenceBasedMethod

class FriedmanHStatisticMethod(PartialDependenceBasedMethod):
    """Class implementing H-statistic for extraction of interactions. 

    Attributes:
        method (str) -- name of interaction method
        visualizer (Visualizer) -- automatically created on the basis of a method and used to create visualizations
        variable_importance (pd.DataFrame) -- variable importance values 
        ovo (pd.DataFrame) -- one versus one variable interaction values 
        ova (pd.DataFrame) -- one vs all feature interactions
        normalized (bool) -- flag determining whether to normalize the interaction values (unnrormalized version is proposed in https://www.tandfonline.com/doi/full/10.1080/10618600.2021.2007935)

    References:
    - https://www.jstor.org/stable/pdf/30245114.pdf
    - https://www.tandfonline.com/doi/full/10.1080/10618600.2021.2007935
    """

    def __init__(self, random_state: Optional[int] = None, normalized: bool = True):
        """Constructor for FriedmanHStatisticMethod

        Parameters:
            normalized (bool, optional) -- flag determining whether to normalize the interaction values. Defaults to True.
            random_state (int, optional) -- random state for reproducibility. Defaults to None.
        """
        super().__init__(InteractionMethod.H_STATISTIC, random_state=random_state)
        self.ova = None
        self.normalized = normalized
        self._pdp_cache = dict()

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False):
        """Calculates H-statistic Interactions and Partial Dependence Based Importance for the given model. 
        Despite pair interactions, this method also calculates one vs all interactions.

        Parameters:
            model -- model to be explained
            X (pd.DataFrame, optional) -- data used to calculate interactions
            n (int, optional) -- number of samples to be used for calculation of interactions
            features (List[str], optional) -- list of features for which interactions will be calculated
            show_progress (bool) -- whether to show progress bar 
        """
        super().fit(model, X, n, features, show_progress, self._pdp_cache)
        self.ova = self._ova(self.predict_function, self.model, self.X_sampled, show_progress, self.features_included)

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """Plots interactions
        
        Parameters:
            vis_type (str) -- type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'summary', 'bar_chart_ova']
            title (str) -- title of plot, default is 'default' which means that title will be automatically generated for selected visualization type
            figsize (tuple) -- size of figure
            show (bool) -- whether to show plot
            **kwargs: additional arguments for plot 
        """
        if self.ova is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             self.ova,
                             variable_importance=self.variable_importance,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self.interactions_ascending_order,
                             **kwargs)

    def _ova(self, predict_function, model, X: pd.DataFrame, progress: bool, features: List[str]) -> pd.DataFrame:
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
            [column, self._calculate_i_versus(predict_function, model, X, column, remove_element(X.columns, column))]
            for column in tqdm(features, desc=ProgressInfoLog.CALC_OVA, disable=not progress)
        ]

        return pd.DataFrame(h_stat_one_vs_all, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=self.interactions_ascending_order, ignore_index=True
        )

    def _calculate_i_versus(self, predict_function, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        """Friedmann H-statistic feature interaction specifics can be found in https://arxiv.org/pdf/0811.1679.pdf"""
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
        return np.sum(nominator) / np.sum(denominator) if self.normalized else np.sqrt(np.sum(nominator))


def _pdp_cache_key(column, row):
    return column, row[column]


def _take_from_cache_or_calc(pdp_cache, key, X_sampled, change_dict, predict_function, model):
    return pdp_cache[key] if key in pdp_cache else partial_dependence_value(X_sampled, change_dict, predict_function, model)
