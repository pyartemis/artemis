from statistics import stdev
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.util.partial_dependence import partial_dependence_value
from src.util.ops import sample_if_not_none


def calculate_vint(model, X: pd.DataFrame, n: int = None, show_progress: bool = False) -> pd.DataFrame:
    X_sampled = sample_if_not_none(X, n)
    pairs = list(combinations(X_sampled.columns, 2))
    vint_pairs = [
        [c1, c2, calculate_vint_ij(model, X_sampled, c1, c2)]
        for c1, c2 in tqdm(pairs, disable=not show_progress)
    ]

    return pd.DataFrame(vint_pairs, columns=["Feature 1", "Feature 2", "Variable Interaction"]).sort_values(
        by="Variable Interaction", ascending=False, ignore_index=True
    )


def calculate_vint_ij(model, X: pd.DataFrame, i: str, j: str) -> float:
    pd_values = np.array(
        [
            [partial_dependence_value(X, {i: x_i, j: x_j}, model.predict) for x_i in set(X[i])]
            for x_j in set(X[j])
        ]
    )
    res_j = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 1, pd_values))
    res_i = np.apply_along_axis(stdev, 0, np.apply_along_axis(stdev, 0, pd_values))
    return (res_j + res_i) / 2
