from abc import abstractmethod, ABC
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from artemis._utilities.domain import VisualizationType
from artemis._utilities.exceptions import MethodNotFittedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from artemis.visualizer._visualizer import Visualizer


class FeatureInteractionMethod(ABC):
    """
    Abstract base class for Feature Interaction Extraction methods. 
    This class should not be used directly. Use derived classes instead.
    
    Attributes
    ----------
    method : str 
        Method name, used also for naming column with results in `results` pd.DataFrame.
    visualizer : Visualizer
        Object providing visualization. Automatically created on the basis of a method and used to create visualizations.
    ovo : pd.DataFrame 
        One versus one (pair) feature interaction values. 
    feature_importance : pd.DataFrame 
        Feature importance values.
    model : object
        Explained model.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.
    """
    def __init__(self, method: str, random_state: Optional[int] = None):
        self.method = method
        self.visualizer = Visualizer(method, VisualizationConfigurationProvider.get(method))
        self.feature_importance = None
        self._feature_importance_obj = None
        self.ovo = None
        self.model = None
        self.X_sampled = None
        self.features_included = None
        self.pairs = None
        self._random_generator = np.random.default_rng(random_state)

    @property
    @abstractmethod
    def _interactions_ascending_order(self):
        ...

    @property
    def _compare_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        return self.ovo.sort_values(self.method, ascending=self._interactions_ascending_order, ignore_index=True)

    @abstractmethod
    def fit(self, model, **kwargs):
        """
        Base abstract method for calculating feature interaction values.

        Parameters
        ----------
        model : object
            Model for which interactions will be extracted. 
        **kwargs : dict
            Parameters specific to a given feature interaction method.
        """
        ...

    def plot(self,
             vis_type: str = VisualizationType.HEATMAP,
             title: str = "default",
             figsize: Tuple[float, float] = (8, 6),
             **kwargs):
        """
        Plot results of explanations.

        There are four types of plots available:
        - heatmap - heatmap of feature interactions values with feature importance values on the diagonal (default)
        - bar_chart - bar chart of top feature interactions values
        - graph - graph of feature interactions values
        - summary - combination of other plots 
        
        Parameters
        ----------
        vis_type : str 
            Type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'summary']. Default is 'heatmap'.
        title : str 
            Title of plot, default is 'default' which means that title will be automatically generated for selected visualization type.
        figsize : (float, float) 
            Size of plot. Default is (8, 6).
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
            Used for 'bar_chart' visualization. Maximum number of pairs that will be presented in plot. Default is 10.
        color : str 
            Used for 'bar_chart' visualization. Color of bars. Default is 'mediumpurple'.

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
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             feature_importance=self.feature_importance,
                             title=title,
                             figsize=figsize,
                             interactions_ascending_order=self._interactions_ascending_order,
                             importance_ascending_order=self._feature_importance_obj.importance_ascending_order,
                             **kwargs)

    def interaction_value(self, f1: str, f2: str):

        if self._compare_ovo is None:
            raise MethodNotFittedException(self.method)

        return self._compare_ovo[((self._compare_ovo["Feature 1"] == f1) & (self._compare_ovo["Feature 2"] == f2)) |
                                 ((self._compare_ovo["Feature 1"] == f2) &
                                  (self._compare_ovo["Feature 2"] == f1))][self.method].values[0]
