from abc import abstractmethod

import pandas as pd

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

    def __init__(self, method: str):
        self.method = method
        self.visualisation = Visualizator(method, VisualisationConfigurationProvider.get(method))
        self.variable_importance = None
        self.ovo = None
        self.X_sampled = None
        self.features_included = None

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

    def plot(self, vis_type: str = VisualisationType.HEATMAP, figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """
        Base method for creating requested type of plot. Can be used only after `fit` method.

        Args:
            vis_type: str, {"summary", "graph", "bar chart", "heatmap"} visualisation type, default "summary"

        Returns:
            object: None
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualisation.plot(self.ovo, vis_type, variable_importance=self.variable_importance, figsize=figsize, show=show, kwargs=kwargs)
       

    def interaction_value(self, f1: str, f2: str):

        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        return self.ovo[
            ((self.ovo["Feature 1"] == f1) & (self.ovo["Feature 2"] == f2)) |
            ((self.ovo["Feature 1"] == f2) & (self.ovo["Feature 2"] == f1))
            ][self.method].values[0]

    def sorted_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        return self.ovo.sort_values(by=self.method, ascending=False, ignore_index=True)
