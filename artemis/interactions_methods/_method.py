from abc import abstractmethod, ABC

import pandas as pd

from artemis.utilities.domain import VisualizationType
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from artemis.visualizer._visualizer import Visualizer


class FeatureInteractionMethod(ABC):
    """Abstract base class for interaction methods. This class should not be used directly. Use derived classes instead.

        Attributes:
            method              [str], name of interaction method
            visualizer          [Visualizer], automatically created on the basis of a method and used to create visualizations
            variable_importance [pd.DataFrame], object that stores variable importance values after fitting
            ovo                 [pd.DataFrame], stores one versus one variable interaction values after fitting
            X_sampled           [pd.DataFrame], data used to calculate interactions
            features_included   [List[str]], list of features that will be used during interactions calculation, if None is passed, all features will be used
    """

    def __init__(self, method: str):
        self.method = method
        self.visualizer = Visualizer(method, VisualizationConfigurationProvider.get(method))
        self.variable_importance = None
        self.ovo = None
        self.X_sampled = None
        self.features_included = None

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
    def fit(self, model, X: pd.DataFrame, **kwargs):
        """
        Base abstract method for calculating feature interaction method values.

        Args:
            model:  model for which interactions will be extracted. Must have implemented `predict` method
            X:  data used to calculate interactions
            **kwargs:   parameters specific to a given feature interaction method

        Returns:
            object: None
        """
        ...

    def plot(self, vis_type: str = VisualizationType.HEATMAP, figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """
        Base method for creating requested type of plot. Can be used only after `fit` method.

        Args:
            vis_type: str, {"summary", "graph", "bar chart", "heatmap"} visualizer type, default "summary"

        Returns:
            object: None
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo, vis_type, variable_importance=self.variable_importance, figsize=figsize, show=show, **kwargs)


    def interaction_value(self, f1: str, f2: str):

        if self._compare_ovo is None:
            raise MethodNotFittedException(self.method)

        return self._compare_ovo[
            ((self._compare_ovo["Feature 1"] == f1) & (self._compare_ovo["Feature 2"] == f2)) |
            ((self._compare_ovo["Feature 1"] == f2) & (self._compare_ovo["Feature 2"] == f1))
            ][self.method].values[0]
