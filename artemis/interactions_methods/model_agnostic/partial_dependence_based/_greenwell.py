from statistics import stdev
from typing import List

import numpy as np
import pandas as pd
from artemis.utilities.domain import InteractionMethod
from artemis.utilities.ops import partial_dependence_value
from ._pdp import PartialDependenceBasedMethod


class GreenwellMethod(PartialDependenceBasedMethod):

    def __init__(self):
        super().__init__(InteractionMethod.VARIABLE_INTERACTION)

    def _calculate_i_versus(self, predict_function, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        j = versus[0]  # only OvO
        pd_values = np.array(
            [
                [partial_dependence_value(X_sampled, {i: x_i, j: x_j},
                                                                       predict_function, model) for x_i in
                 set(X_sampled[i])]
                for x_j in set(X_sampled[j])
            ]
        )
        res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
        res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
        return (res_j + res_i) / 2
