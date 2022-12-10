from statistics import stdev
from typing import List, Optional

import numpy as np
import pandas as pd

from artemis.utilities.domain import InteractionMethod
from artemis.utilities.ops import partial_dependence_value
from ._pdp import PartialDependenceBasedMethod


class GreenwellMethod(PartialDependenceBasedMethod):
    """Class implementing Greenwell feature interaction method.

    Attributes:
        method (str) -- name of interaction method
        visualizer (Visualizer) -- automatically created on the basis of a method and used to create visualizations
        variable_importance (pd.DataFrame) -- variable importance values 
        ovo (pd.DataFrame) -- one versus one variable interaction values 
    
    References:
    - https://arxiv.org/pdf/1805.04755.pdf
    """


    def __init__(self, random_state: Optional[int] = None):
        """Constructor for GreenwellMethod
        
        Parameters:
            random_state (int, optional) -- random state for reproducibility. Defaults to None."""
        super().__init__(InteractionMethod.VARIABLE_INTERACTION, random_state=random_state)

    def _calculate_i_versus(self, predict_function, model, X_sampled: pd.DataFrame, i: str, versus: List[str]) -> float:
        """
        See `_calculate_i_versus` documentation in `PartialDependenceBasedMethod`.
        Average of standard deviation over PDP profile for i-th, and j-th feature.
        Specifics can be found in: https://arxiv.org/pdf/1805.04755.pdf
        """
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
