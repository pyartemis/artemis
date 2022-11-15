from abc import abstractmethod

import pandas as pd

from artemis.utilities.domain import VisualisationType
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.visualisation.configuration import VisualisationConfigurationProvider
from artemis.visualisation.visualisation import Visualisation


class FeatureInteractionMethod:
    def __init__(self, method: str):
        self.method = method
        self.visualisation = Visualisation(method, VisualisationConfigurationProvider.get(method))
        self.ovo = None
        self.X_sampled = None
        self.features_included = None

    @abstractmethod
    def fit(self, model, X: pd.DataFrame, **kwargs):
        ...

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualisation.plot(self.ovo, vis_type)

    def interaction_value(self, f1, f2):

        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        return self.ovo[
            ((self.ovo["Feature 1"] == f1) & (self.ovo["Feature 2"] == f2)) |
            ((self.ovo["Feature 1"] == f2) & (self.ovo["Feature 2"] == f1))
            ][self.method].values[0]
