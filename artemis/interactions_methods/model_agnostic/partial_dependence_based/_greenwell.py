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


    def _ovo(self, predict_function, model, X_sampled: pd.DataFrame, show_progress: bool, batchsize: int):
        full_pd = _calculate_pdp_with_batch(model, predict_function, X_sampled, show_progress, self.pairs, batchsize)
        value_pairs = []
        for pair, pd_raw_values in full_pd.groupby(["Feature 1", "Feature 2"]):
            pd_values = pd.crosstab(pd_raw_values["F1 value"], pd_raw_values["F2 value"], values=pd_raw_values["PD"], aggfunc=np.mean)
            res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
            res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
            value_pairs.append([pair[0], pair[1], (res_j + res_i) / 2])
        return pd.DataFrame(value_pairs, columns=["Feature 1", "Feature 2", self.method]).sort_values(
            by=self.method, ascending=self.interactions_ascending_order, ignore_index=True
        ).fillna(0)

def _calculate_pdp_with_batch(model, predict_function, X_sampled, show_progress, pairs, batchsize):
    X_sampled_len = len(X_sampled)
    pdp_dict = dict()
    current_len = 0
    range_dict = {}
    X_full = pd.DataFrame()

    for i, j in tqdm(pairs, desc=ProgressInfoLog.CALC_OVO, disable=not show_progress):
        for x_i in set(X_sampled[i]):
            for x_j in set(X_sampled[j]):
                change_dict = {i: x_i, j: x_j}
                X_changed = X_sampled.assign(**change_dict)
                range_dict[(i, j, x_i, x_j)] = (current_len, current_len + len(X_changed))
                current_len += X_sampled_len
                X_full = pd.concat((X_full, X_changed))
            if current_len > batchsize:
                pdp_dict = _add_mean_preds_to_pdp_dict(model, predict_function, X_full, range_dict, pdp_dict)
                current_len = 0
                range_dict = {}
                X_full = pd.DataFrame()
    if current_len > 0:
        pdp_dict = _add_mean_preds_to_pdp_dict(model, predict_function, X_full, range_dict, pdp_dict)
        current_len = 0
        range_dict = {}
        X_full = pd.DataFrame()

    res = pd.DataFrame.from_dict(pdp_dict, orient='index', columns=['PD'])
    res.index = pd.MultiIndex.from_tuples(tuples=res.index, names=['Feature 1', 'Feature 2', 'F1 value', 'F2 value'])
    res = res.reset_index()
    return res

def _add_mean_preds_to_pdp_dict(model, predict_function, X_full, range_dict, pdp_dict):
    y = predict_function(model, X_full)
    for i, j, x_i, x_j in range_dict.keys():
        start, end = range_dict[(i, j, x_i, x_j)]
        pdp_dict[(i, j, x_i, x_j)] = np.mean(y[start:end])
    return pdp_dict