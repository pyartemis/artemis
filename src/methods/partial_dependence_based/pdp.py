from abc import abstractmethod
from itertools import combinations
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from src.domain.domain import VisualisationType, SUMMARY, ONE_VS_ONE, INTERACTION_GRAPH, BAR_CHART, HEATMAP
from src.util.exceptions import VisualisationNotSupportedException
from src.visualisation.visualisation import Visualisation


class PartialDependenceBasedMethod:

    def __init__(self, method: str, accepted_visualisations: List[VisualisationType]):
        self.accepted_visualisations = accepted_visualisations
        self.ovo = None
        self.method = method

    def fit_(self,
             model,
             X_sampled: pd.DataFrame,
             features_included: List[str] = None,
             show_progress: bool = False):
        self.ovo = self._ovo(model, X_sampled, show_progress, features_included)

    def plot_(self, ova: pd.DataFrame = None, vis_type: VisualisationType = SUMMARY):

        if vis_type not in self.accepted_visualisations:
            raise VisualisationNotSupportedException(self.method, vis_type)

        vis = Visualisation(method=self.method)

        if vis_type == SUMMARY:
            vis.plot_summary(self.ovo, ova)
        elif vis_type == INTERACTION_GRAPH:
            vis.plot_interaction_graph(self.ovo)
        elif vis_type == BAR_CHART:
            vis.plot_barchart(ova)
        elif vis_type == HEATMAP:
            vis.plot_heatmap(self.ovo)

    def _ovo(self, model, X_sampled: pd.DataFrame, show_progress: bool, features: List[str]):
        pairs = list(combinations(features, 2))
        h_stat_pairs = [
            [c1, c2, self._calculate_i_versus(model, X_sampled, c1, [c2])]
            for c1, c2 in tqdm(pairs, desc=ONE_VS_ONE.value, disable=not show_progress)
        ]

        return pd.DataFrame(h_stat_pairs, columns=["Feature 1", "Feature 2", self.method.__str__()]).sort_values(
            by=self.method.__str__(), ascending=False, ignore_index=True
        )

    def partial_dependence_value(self, df: pd.DataFrame, change_dict: Dict, predict_function: Callable) -> ndarray:
        assert all(column in df.columns for column in change_dict.keys())
        df_changed = df.assign(**change_dict)
        return np.mean(predict_function(df_changed))

    @abstractmethod
    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        ...
