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
            random_state (int, optional) -- random state for reproducibility. Defaults to None.
            normalized (bool, optional) -- flag determining whether to normalize the interaction values. Defaults to True.
        """
        super().__init__(InteractionMethod.H_STATISTIC, random_state=random_state)
        self.ova = None
        self.normalized = normalized

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False,
            calculate_ova: bool = True):
        """Calculates H-statistic Interactions and Partial Dependence Based Importance for the given model. 
        Despite pair interactions, this method also calculates one vs all interactions.

        Parameters:
            model -- model to be explained
            X (pd.DataFrame, optional) -- data used to calculate interactions
            n (int, optional) -- number of samples to be used for calculation of interactions
            features (List[str], optional) -- list of features for which interactions will be calculated
            show_progress (bool) -- whether to show progress bar 
        """
        super().fit(model, X, n, features, show_progress)
        if calculate_ova:
            self.ova = self._calculate_ova_interactions_from_pdp(show_progress)

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
                             importance_ascending_order=self._variable_importance_obj.importance_ascending_order,
                             **kwargs)

    def _calculate_ova_interactions_from_pdp(self, show_progress: bool) -> pd.DataFrame:
        self.pdp_calculator.calculate_pd_minus_single(self.features_included, show_progress=show_progress)
        preds = self.predict_function(self.model, self.X_sampled)
        value_minus_single = []
        for feature in self.features_included:
            pd_f = self.pdp_calculator.get_pd_single(feature, feature_values = self.X_sampled[feature].values)
            pd_f_minus = self.pdp_calculator.get_pd_minus_single(feature)
            value_minus_single.append([feature, _calculate_hstat_value(pd_f, pd_f_minus, preds, self.normalized)])
        return pd.DataFrame(value_minus_single, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=self.interactions_ascending_order, ignore_index=True
        ).fillna(0)

    def _calculate_ovo_interactions_from_pdp(self, show_progress: bool):
        self.pdp_calculator.calculate_pd_pairs(self.pairs, show_progress=show_progress, all_combinations=False)
        self.pdp_calculator.calculate_pd_single(self.features_included, show_progress=False)
        value_pairs = []
        for pair in self.pairs:
            pd_f1 = self.pdp_calculator.get_pd_single(pair[0], feature_values = self.X_sampled[pair[0]].values)
            pd_f2 = self.pdp_calculator.get_pd_single(pair[1], feature_values = self.X_sampled[pair[1]].values)
            pair_feature_values = list(zip(self.X_sampled[pair[0]].values, self.X_sampled[pair[1]].values))
            pd_pair = self.pdp_calculator.get_pd_pairs(pair[0], pair[1], feature_values = pair_feature_values)
            value_pairs.append([pair[0], pair[1], _calculate_hstat_value(pd_f1, pd_f2, pd_pair, self.normalized)])
        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=self.interactions_ascending_order, ignore_index=True
        ).fillna(0)

def _calculate_hstat_value(pd_f1: np.ndarray, pd_f2: np.ndarray, pd_pair: np.ndarray, normalized: bool = True):
    nominator = (center(pd_pair) - center(pd_f1) - center(pd_f2)) ** 2
    if normalized: 
        denominator = center(pd_pair) ** 2
    return np.sum(nominator) / np.sum(denominator) if normalized else np.sqrt(np.sum(nominator))
