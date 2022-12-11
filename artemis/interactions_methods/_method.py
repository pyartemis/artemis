from abc import abstractmethod, ABC
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from artemis.utilities.domain import VisualizationType
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from artemis.visualizer._visualizer import Visualizer


class FeatureInteractionMethod(ABC):
    """
    Abstract base class for Feature Interaction Extraction methods. 
    This class should not be used directly. Use derived classes instead.
    
    Attributes:
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
    def interactions_ascending_order(self):
        ...

    @property
    def _compare_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        return self.ovo.sort_values(self.method, ascending=self.interactions_ascending_order, ignore_index=True)

    @abstractmethod
    def fit(self, model, **kwargs):
        """
        Base abstract method for calculating feature interaction values.

        Parameters:
        ----------
        model : object
            Model for which interactions will be extracted. 
        **kwargs : dict
            Parameters specific to a given feature interaction method.
        """
        ...

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: Tuple[float, float] = (8, 6), show: bool = True, **kwargs):
        """
        Plot results of explanations.

        There are four types of plots available:
        - heatmap - heatmap of feature interactions values with feature importance values on the diagonal (default)
        - bar_chart - bar chart of top feature interactions values
        - graph - graph of feature interactions values
        - summary - combination of other plots 
        
        Parameters:
        ----------
        vis_type : str 
            Type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'summary']. Default is 'heatmap'.
        title : str 
            Title of plot, default is 'default' which means that title will be automatically generated for selected visualization type.
        figsize : (float, float) 
            Size of plot. Default is (8, 6).
        show : bool 
            Whether to show plot. Default is True.
        **kwargs : dict
            Additional arguments for plot.
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             feature_importance=self.feature_importance,
                             title=title,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self.interactions_ascending_order,
                             importance_ascending_order=self._feature_importance_obj.importance_ascending_order,
                             **kwargs)


    def interaction_value(self, f1: str, f2: str):

        if self._compare_ovo is None:
            raise MethodNotFittedException(self.method)

        return self._compare_ovo[
            ((self._compare_ovo["Feature 1"] == f1) & (self._compare_ovo["Feature 2"] == f2)) |
            ((self._compare_ovo["Feature 1"] == f2) & (self._compare_ovo["Feature 2"] == f1))
            ][self.method].values[0]
