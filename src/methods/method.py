from abc import abstractmethod

import pandas as pd

from src.domain.domain import VisualisationType
from src.util.exceptions import VisualisationNotSupportedException
from src.visualisation.visualisation import Visualisation


class FeatureInteractionMethod:
    def __init__(self, accepted_visualisations, method):
        self.method = method
        self.accepted_visualisations = accepted_visualisations
        self.ovo = None
        self.X_sampled = None
        self.features_included = None

    @abstractmethod
    def fit(self, model, X: pd.DataFrame, **kwargs):
        ...

    def plot(self, vis_type: VisualisationType = VisualisationType.SUMMARY):
        assert self.ovo is not None, "Before executing plot() method, fit() must be executed!"
        self.plot_(Visualisation(method=self.method), vis_type=vis_type)

    def plot_(self,
              vis: Visualisation,
              ova: pd.DataFrame = None,
              vis_type: VisualisationType = VisualisationType.SUMMARY):

        if vis_type not in self.accepted_visualisations:
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            vis.plot_summary(self.ovo, ova)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            vis.plot_interaction_graph(self.ovo)
        elif vis_type == VisualisationType.BAR_CHART:
            vis.plot_barchart(ova)
        elif vis_type == VisualisationType.HEATMAP:
            vis.plot_heatmap(self.ovo)
