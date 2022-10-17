from typing import List

import pandas as pd
from src.util.partial_dependence import partial_dependence_value
from tqdm import tqdm


"""

TODO:
1. Check correctness of H-statistic calculation
2. Wrap around more usable (less generic) function providing ability to create OvA and OvO summaries for all features
3. Provide unit-tests for H-statistic

"""


def calculate_h_stat_i_versus(model, X: pd.DataFrame, i: str, versus: List[str]) -> float:
    nominator = denominator = 0
    for _, row in tqdm(X.iterrows()):
        change_i = {i: row[i]}
        change_versus = {col: row[col] for col in versus}
        change_i_versus = {**change_i, **change_versus}

        pd_i = partial_dependence_value(X, change_i, model.predict)
        pd_versus = partial_dependence_value(X, change_versus, model.predict)
        pd_i_versus = partial_dependence_value(X, change_i_versus, model.predict)

        nominator += (pd_i_versus - pd_i - pd_versus) ** 2
        denominator += pd_i_versus**2

    return nominator / denominator
