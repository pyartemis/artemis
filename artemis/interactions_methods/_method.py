from abc import abstractmethod, ABC
from typing import Callable, Optional

import numpy as np
import pandas as pd

from artemis.utilities.domain import VisualizationType
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from artemis.visualizer._visualizer import Visualizer


class FeatureInteractionMethod(ABC):
    """Abstract base class for interaction methods. This class should not be used directly. Use derived classes instead.
    Attributes:
        method  (str) -- name of interaction method
        visualizer (Visualizer) -- automatically created on the basis of a method and used to create visualizations
        feature_importance (pd.DataFrame) -- variable importance values 
        ovo (pd.DataFrame) -- one versus one variable interaction values 
        X_sampled (pd.DataFrame) -- data used to calculate interactions
        features_included  (List[str]) -- list of features that will be used during interactions calculation, if None is passed, all features will be used
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
        self.random_state = random_state
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
