from statistics import stdev
from typing import List, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm
from artemis.utilities.domain import InteractionMethod, ProgressInfoLog
from artemis.utilities.ops import partial_dependence_value
from ._pdp import PartialDependenceBasedMethod


class GreenwellMethod(PartialDependenceBasedMethod):
    """
    Greenwell Method for Feature Interaction Extraction. 
    
    Uses partial dependence values to calculate variable interaction strengths and variable importance. 

    Attributes:
    ----------
    method : str 
        Method name, used also for naming column with results in `results` pd.DataFrame.
    visualizer : Visualizer
        Object providing visualization. Automatically created on the basis of a method and used to create visualizations.
    ovo : pd.DataFrame 
        One versus one (pair) feature interaction values. 
    feature_importance : pd.DataFrame 
        Feature importance values.
    model : object
        Explained model.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.
    pd_calculator : PartialDependenceCalculator
        Object used to calculate and store partial dependence values.
    batchsize: int
        Batch size used for calculation.

    References:
    ----------
    - https://arxiv.org/pdf/1805.04755.pdf
    """


    def __init__(self, random_state: Optional[int] = None):
        """Constructor for GreenwellMethod
        
        Parameters:
        ----------
        random_state : int, optional 
            Random state for reproducibility. Defaults to None.
        """
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

