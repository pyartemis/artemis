from statistics import stdev
from typing import List, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm
from artemis.utilities.domain import InteractionMethod, ProgressInfoLog
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

    def _calculate_ovo_interactions_from_pd(self, show_progress: bool = False):
        self.pd_calculator.calculate_pd_pairs(self.pairs, show_progress=show_progress)
        value_pairs = []
        for pair in self.pairs:
            pd_values = self.pd_calculator.get_pd_pairs(pair[0], pair[1])
            res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
            res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
            value_pairs.append([pair[0], pair[1], (res_j + res_i) / 2])
        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=self.interactions_ascending_order, ignore_index=True
        ).fillna(0)

