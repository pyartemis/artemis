from abc import abstractmethod
from itertools import combinations
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from src.domain.domain import VisualisationType, InteractionCalculationStrategy
from src.util.exceptions import VisualisationNotSupportedException
from src.util.ops import sample_if_not_none, all_if_none
from src.visualisation.visualisation import Visualisation


class PartialDependenceBasedMethod:

    def __init__(self, method: str, accepted_visualisations: List[str]):
        self.accepted_visualisations = accepted_visualisations
        self.ovo = None
        self.X_sampled = None
        self.features_included = None
        self.method = method

    def fit_(self,
             model,
             X: pd.DataFrame,
             n: int = None,
             features: List[str] = None,
             show_progress: bool = False):
        self.X_sampled = sample_if_not_none(X, n)
        self.features_included = all_if_none(X, features)

        self.ovo = self._ovo(model, self.X_sampled, show_progress, self.features_included)

    def plot_(self, ova: pd.DataFrame = None, vis_type: VisualisationType = VisualisationType.SUMMARY):

        if vis_type not in self.accepted_visualisations:
            raise VisualisationNotSupportedException(self.method, vis_type)

        vis = Visualisation(method=self.method)

        if vis_type == VisualisationType.SUMMARY:
            vis.plot_summary(self.ovo, ova)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            vis.plot_interaction_graph(self.ovo)
        elif vis_type == VisualisationType.BAR_CHART:
            vis.plot_barchart(ova)
        elif vis_type == VisualisationType.HEATMAP:
            vis.plot_heatmap(self.ovo)

    def _ovo(self, model, X_sampled: pd.DataFrame, show_progress: bool, features: List[str]):
        pairs = list(combinations(features, 2))
        value_pairs = [
            [c1, c2, self._calculate_i_versus(model, X_sampled, c1, [c2])]
            for c1, c2 in tqdm(pairs, desc=InteractionCalculationStrategy.ONE_VS_ONE, disable=not show_progress)
        ]

        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=False, ignore_index=True
        )

    @staticmethod
    def partial_dependence_value(df: pd.DataFrame, change_dict: Dict, predict_function: Callable) -> ndarray:
        assert all(column in df.columns for column in change_dict.keys())
        df_changed = df.assign(**change_dict)
        return np.mean(predict_function(df_changed))

    @abstractmethod
    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        ...
