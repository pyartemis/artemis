from statistics import stdev
from typing import List

import numpy as np
import pandas as pd
from src.domain.domain import SUMMARY, VisualisationType, INTERACTION_GRAPH, HEATMAP, Method
from src.methods.partial_dependence_based.pdp import PartialDependenceBasedMethod
from src.util.ops import sample_if_not_none, all_if_none


class GreenwellVariableInteraction(PartialDependenceBasedMethod):

    def __init__(self):
        super().__init__(Method.VARIABLE_INTERACTION, [SUMMARY, INTERACTION_GRAPH, HEATMAP])

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False):
        X_sampled = sample_if_not_none(X, n)
        features_included = all_if_none(X, features)
        super().fit_(model, X_sampled, features_included, show_progress)

    def plot(self, vis_type: VisualisationType = SUMMARY):
        assert self.ovo is not None, "Before executing plot() method, fit() must be executed!"
        super().plot_(vis_type=vis_type)

    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        j = versus[0]  # only OvO
        pd_values = np.array(
            [
                [super(GreenwellVariableInteraction, self).partial_dependence_value(X_sampled, {i: x_i, j: x_j},
                                                                                    model.predict) for x_i in
                 set(X_sampled[i])]
                for x_j in set(X_sampled[j])
            ]
        )
        res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
        res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
        return (res_j + res_i) / 2
