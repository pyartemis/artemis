from typing import List

import pandas as pd
from src.util.partial_dependence import find_index, partial_dependence_i_versus

"""

TODO:
1. Check correctness of H-statistic calculation
2. Find more efficient way of calculating partial dependence profiles for OvO and OvA
3. Wrap around more usable (less generic) function providing ability to create OvA and OvO summaries for all features
4. Provide unit-tests for H-statistic

"""


# naive implementation of h statistic calculation (extremely slow)
def calculate_h_stat_i_versus(model, X: pd.DataFrame, i: int, versus: List[int], class_number: int = 0) -> float:
    pd_i, pd_versus, pd_i_versus = partial_dependence_i_versus(model, X, i, versus)

    nominator = denominator = 0
    for index, row in X.iterrows():
        i_index = find_index(pd_i_versus, 0, row[i])
        versus_indexes = [find_index(pd_i_versus, column + 1, row[versus[column]]) for column in range(len(versus))]

        pd_i_versus_index = pd_i_versus['average'][class_number][i_index][tuple(versus_indexes)]
        pd_i_index = pd_i['average'][class_number][i_index]
        pd_versus_index = pd_versus['average'][class_number][tuple(versus_indexes)]

        nominator += (pd_i_versus_index - pd_i_index - pd_versus_index) ** 2
        denominator += pd_i_versus_index ** 2

    return nominator / denominator
