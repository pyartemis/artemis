from statistics import stdev
from typing import List

import numpy as np
import pandas as pd
from src.domain.domain import VisualisationType, Method
from src.methods.partial_dependence_based.pdp import PartialDependenceBasedMethod


class GreenwellVariableInteraction(PartialDependenceBasedMethod):

    def __init__(self):
        super().__init__(Method.VARIABLE_INTERACTION,
                         [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH, VisualisationType.HEATMAP])

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False):
        super().sample_ovo(model, X, n, features, show_progress)

    def _calculate_i_versus(self, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        j = versus[0]  # only OvO
        pd_values = np.array(
            [
                [PartialDependenceBasedMethod.partial_dependence_value(X_sampled, {i: x_i, j: x_j},
                                                                       model.predict) for x_i in
                 set(X_sampled[i])]
                for x_j in set(X_sampled[j])
            ]
        )
        res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
        res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
        return (res_j + res_i) / 2
