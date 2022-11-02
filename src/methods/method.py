from abc import abstractmethod

import pandas as pd

from src.domain.domain import VisualisationType
from src.visualisation.configuration import VisualisationConfigurationProvider
from src.visualisation.visualisation import Visualisation


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
        assert self.ovo is not None, "Before executing plot() method, fit() must be executed!"
        self.visualisation.plot(self.ovo, vis_type)
