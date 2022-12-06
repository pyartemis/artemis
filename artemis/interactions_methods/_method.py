from abc import abstractmethod

import pandas as pd
import numpy as np

from artemis.utilities.domain import VisualisationType
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.visualisation.configuration import VisualisationConfigurationProvider
from artemis.visualisation.visualisator import Visualizator


class FeatureInteractionMethod:
    """Abstract base class for interaction methods. This class should not be used directly. Use derived classes instead.

        Attributes:
            method              [str], name of interaction method
            visualisation       [Visualisation], automatically created on the basis of a method and used to create visualisations
            variable_importance [pd.DataFrame], object that stores variable importance values after fitting
            ovo                 [pd.DataFrame], stores one versus one variable interaction values after fitting
            X_sampled           [pd.DataFrame], data used to calculate interactions
            features_included   [List[str]], list of features that will be used during interactions calculation, if None is passed, all features will be used
    """

    def __init__(self, method: str, random_state: Optional[int] = None):
        self.method = method
        self.visualisation = Visualizator(method, VisualisationConfigurationProvider.get(method))
        self.variable_importance = None
        self.ovo = None
        self.X_sampled = None
        self.features_included = None
        self.random_state = random_state
        self.random_generator = np.random.default_rng(random_state)

    @property
    @abstractmethod
    def interactions_ascending_order(self):
        ...

    @property
    def _compare_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        return self.ovo.sort_values(self.method, ascending=self.interactions_ascending_order, ignore_index=True)

    @property
    def _compare_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        return self.ovo.sort_values(self.method, ascending=self.interactions_ascending_order, ignore_index=True)

    @abstractmethod
    def fit(self, model, X: pd.DataFrame, **kwargs):
        """
        Base abstract method for calculating feature interaction method values.

        Parameters:
            model:  model for which interactions will be extracted. Must have implemented `predict` method
            X:  data used to calculate interactions
            **kwargs:   parameters specific to a given feature interaction method

        Returns:
            object: None
        """
        ...

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """Plots interactions
        
        Parameters:
            vis_type (str) -- type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'summary']
            title (str) -- title of plot, default is 'default' which means that title will be automatically generated for selected visualization type
            figsize (tuple) -- size of figure
            show (bool) -- whether to show plot
            **kwargs: additional arguments for plot 
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             variable_importance=self.variable_importance,
                             title=title,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self.interactions_ascending_order,
                             **kwargs)


    def interaction_value(self, f1: str, f2: str):

        if self._compare_ovo is None:
            raise MethodNotFittedException(self.method)

        return self._compare_ovo[
            ((self._compare_ovo["Feature 1"] == f1) & (self._compare_ovo["Feature 2"] == f2)) |
            ((self._compare_ovo["Feature 1"] == f2) & (self._compare_ovo["Feature 2"] == f1))
            ][self.method].values[0]
