from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis._utilities.domain import VisualizationType, InteractionMethod, ProgressInfoLog
from artemis._utilities.exceptions import MethodNotFittedException
from artemis._utilities.ops import remove_element, center, partial_dependence_value
from artemis._utilities.pd_calculator import PartialDependenceCalculator
from ._pdp import PartialDependenceBasedMethod

class FriedmanHStatisticMethod(PartialDependenceBasedMethod):
    """
    Friedman's H-statistic Method for Feature Interaction Extraction. 
    
    Uses partial dependence values to calculate feature interaction strengths and feature importance. 

    Attributes
    ----------
    method : str 
        Method name, used also for naming column with results in `ovo` pd.DataFrame.
    visualizer : Visualizer
        Object providing visualization. Automatically created on the basis of a method and used to create visualizations.
    ovo : pd.DataFrame 
        One versus one (pair) feature interaction values. 
    feature_importance : pd.DataFrame 
        Feature importance values.
    ova : pd.DataFrame
        One vs all feature interaction values.
    normalized : bool 
        Flag determining whether interaction values are normalized.
        Unnrormalized version is proposed in https://www.tandfonline.com/doi/full/10.1080/10618600.2021.2007935
    model : object
        Explained model.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.
    pd_calculator : PartialDependenceCalculator
        Object used to calculate and store partial dependence values.
    batchsize : int
        Batch size used for calculation.

    References
    ----------
    - https://www.jstor.org/stable/pdf/30245114.pdf
    - https://www.tandfonline.com/doi/full/10.1080/10618600.2021.2007935
    """
    def __init__(self, random_state: Optional[int] = None, normalized: bool = True):
        """Constructor for FriedmanHStatisticMethod

        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility. Defaults to None.
        normalized : bool, optional 
            Flag determining whether to normalize the interaction values. Normalized version is original H-statistic, 
            unnrormalized version is square root of nominator of H statistic. Defaults to True which translates to original H-statistic.
        """
        super().__init__(InteractionMethod.H_STATISTIC, random_state=random_state)
        self.ova = None
        self.normalized = normalized

    def fit(self,
            model,
            X: pd.DataFrame,
            n: Optional[int] = None,
            predict_function: Optional[Callable] = None,
            features: Optional[List[str]] = None,
            show_progress: bool = False,
            batchsize: int = 2000,
            pd_calculator: Optional[PartialDependenceCalculator] = None,
            calculate_ova: bool = True):
        """Calculates H-statistic Feature Interactions Strength and Feature Importance for the given model. 
        Despite pair interactions, this method can also calculate one vs all interactions.

        Parameters
        ----------
        model : object
            Model to be explained, should have predict_proba or predict method, or predict_function should be provided. 
        X : pd.DataFrame
            Data used to calculate interactions. If n is not None, n rows from X will be sampled. 
        n : int, optional
            Number of samples to be used for calculation of interactions. If None, all rows from X will be used. Default is None.
        predict_function : Callable, optional
            Function used to predict model output. It should take model and dataset and outputs predictions. 
            If None, `predict_proba` method will be used if it exists, otherwise `predict` method. Default is None.
        features : List[str], optional
            List of features for which interactions will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        batchsize : int
            Batch size for calculating partial dependence. Prediction requests are collected until the batchsize is exceeded, 
            then the model is queried for predictions jointly for many observations. It speeds up the operation of the method.
            Default is 2000.
        pd_calculator : PartialDependenceCalculator, optional
            PartialDependenceCalculator object containing partial dependence values for a given model and dataset. 
            Providing this object speeds up the calculation as partial dependence values do not need to be recalculated.
            If None, it will be created from scratch. Default is None.
        calculate_ova : bool
            If True, one vs all interactions will be calculated. Default is True.
        """
        super().fit(model, X, n, predict_function, features, show_progress, batchsize, pd_calculator)
        if calculate_ova:
            self.ova = self._calculate_ova_interactions_from_pd(show_progress)

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """
        Plot results of explanations.

        There are five types of plots available:
        - heatmap - heatmap of feature interactions values with feature importance values on the diagonal (default)
        - bar_chart - bar chart of top feature interactions values
        - graph - graph of feature interactions values
        - bar_chart_ova - bar chart of top one vs all interactions values
        - summary - combination of other plots 
        
        Parameters
        ----------
        vis_type : str 
            Type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'bar_chart_ova', 'summary']. Default is 'heatmap'.
        title : str 
            Title of plot, default is 'default' which means that title will be automatically generated for selected visualization type.
        figsize : (float, float) 
            Size of plot. Default is (8, 6).
        show : bool 
            Whether to show plot. Default is True.
        **kwargs : Other Parameters
            Additional parameters for plot. Passed to suitable matplotlib or seaborn functions. 
            For 'summary' visualization parameters for respective plots should be in dict with keys corresponding to visualization name. 
            See key parameters below. 

        Other Parameters
        ------------------------
        interaction_color_map : matplotlib colormap name or object, or list of colors
            Used for 'heatmap' visualization. The mapping from interaction values to color space. Default is 'Purples' or 'Purpler_r',
            depending on whether a greater value means a greater interaction strength or vice versa.
        importance_color_map :  matplotlib colormap name or object, or list of colors
            Used for 'heatmap' visualization. The mapping from importance values to color space. Default is 'Greens' or 'Greens_r',
            depending on whether a greater value means a greater interaction strength or vice versa.
        annot_fmt : str
            Used for 'heatmap' visualization. String formatting code to use when adding annotations with values. Default is '.3f'.
        linewidths : float
            Used for 'heatmap' visualization. Width of the lines that will divide each cell in matrix. Default is 0.5.
        linecolor : str
            Used for 'heatmap' visualization. Color of the lines that will divide each cell in matrix. Default is 'white'.
        cbar_shrink : float
            Used for 'heatmap' visualization. Fraction by which to multiply the size of the colorbar. Default is 1. 
    
        top_k : int 
            Used for 'bar_chart' and 'bar_chart_ova' visualizations. Maximum number of pairs that will be presented in plot. Default is 10.
        color : str 
            Used for 'bar_chart' and 'bar_chart_ova' visualizations. Color of bars. Default is 'mediumpurple'.

        n_highest_with_labels : int
            Used for 'graph' visualization. Top most important interactions to show as labels on edges.  Default is 5.
        edge_color: str
            Used for 'graph' visualization. Color of the edges. Default is 'rebeccapurple.
        node_color: str
            Used for 'graph' visualization. Color of nodes. Default is 'green'.
        node_size: int
            Used for 'graph' visualization. Size of the nodes (networkX scale).  Default is '1800'.
        font_color: str
            Used for 'graph' visualization. Font color. Default is '#3B1F2B'.
        font_weight: str
            Used for 'graph' visualization. Font weight. Default is 'bold'.
        font_size: int
            Used for 'graph' visualization. Font size (networkX scale). Default is 10.
        threshold_relevant_interaction : float
            Used for 'graph' visualization. Minimum (or maximum, depends on method) value of interaction to display
            corresponding edge on visualization. Default depends on the interaction method.
        """
        if self.ova is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             self.ova,
                             feature_importance=self.feature_importance,
                             title=title,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self._interactions_ascending_order,
                             importance_ascending_order=self._feature_importance_obj.importance_ascending_order,
                             **kwargs)

    def _calculate_ova_interactions_from_pd(self, show_progress: bool) -> pd.DataFrame:
        self.pd_calculator.calculate_pd_minus_single(self.features_included, show_progress=show_progress)
        preds = self.predict_function(self.model, self.X_sampled)
        value_minus_single = []
        for feature in self.features_included:
            pd_f = self.pd_calculator.get_pd_single(feature, feature_values = self.X_sampled[feature].values)
            pd_f_minus = self.pd_calculator.get_pd_minus_single(feature)
            value_minus_single.append([feature, _calculate_hstat_value(pd_f, pd_f_minus, preds, self.normalized)])
        return pd.DataFrame(value_minus_single, columns=["Feature", InteractionMethod.H_STATISTIC]).sort_values(
            by=InteractionMethod.H_STATISTIC, ascending=self._interactions_ascending_order, ignore_index=True
        ).fillna(0)

    def _calculate_ovo_interactions_from_pd(self, show_progress: bool):
        self.pd_calculator.calculate_pd_pairs(self.pairs, show_progress=show_progress, all_combinations=False)
        self.pd_calculator.calculate_pd_single(self.features_included, show_progress=False)
        value_pairs = []
        for pair in self.pairs:
            pd_f1 = self.pd_calculator.get_pd_single(pair[0], feature_values = self.X_sampled[pair[0]].values)
            pd_f2 = self.pd_calculator.get_pd_single(pair[1], feature_values = self.X_sampled[pair[1]].values)
            pair_feature_values = list(zip(self.X_sampled[pair[0]].values, self.X_sampled[pair[1]].values))
            pd_pair = self.pd_calculator.get_pd_pairs(pair[0], pair[1], feature_values = pair_feature_values)
            value_pairs.append([pair[0], pair[1], _calculate_hstat_value(pd_f1, pd_f2, pd_pair, self.normalized)])
        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=self._interactions_ascending_order, ignore_index=True
        ).fillna(0)

def _calculate_hstat_value(pd_i: np.ndarray, pd_versus: np.ndarray, pd_i_versus: np.ndarray, normalized: bool = True):
    nominator = (center(pd_i_versus) - center(pd_i) - center(pd_versus)) ** 2
    if normalized: 
        denominator = center(pd_i_versus) ** 2
    return np.sum(nominator) / np.sum(denominator) if normalized else np.sqrt(np.mean(nominator))
