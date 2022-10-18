from itertools import combinations
from typing import List

import pandas as pd
from src.domain.domain import InteractionCalculationStrategy, ONE_VS_ONE, ONE_VS_ALL
from src.util.ops import remove_element, sample_if_not_none
from src.util.partial_dependence import partial_dependence_value
from tqdm import tqdm

"""

TODO:
1. Check correctness of H-statistic calculation
2. Provide unit-tests for H-statistic

"""


def calculate_h_stat(model, X: pd.DataFrame, n: int = None, kind: InteractionCalculationStrategy = ONE_VS_ONE):
    X_sampled = sample_if_not_none(X, n)

    if kind == ONE_VS_ONE:
        return calculate_h_stat_ovo(model, X_sampled)
    elif kind == ONE_VS_ALL:
        return calculate_h_stat_ova(model, X_sampled)
    else:
        raise ValueError(f"Interaction calculation strategy: {kind} unknown.")


def calculate_h_stat_ovo(model, X: pd.DataFrame) -> pd.DataFrame:
    pairs = list(combinations(X.columns, 2))
    h_stat_pairs = [[c1, c2, calculate_h_stat_i_versus(model, X, c1, [c2])] for c1, c2 in pairs if c1 != c2]

    return pd.DataFrame(h_stat_pairs,
                        columns=["Feature 1", "Feature 2", "H-statistic"]).sort_values(by="H-statistic",
                                                                                       ascending=False,
                                                                                       ignore_index=True)


def calculate_h_stat_ova(model, X: pd.DataFrame) -> pd.DataFrame:
    h_stat_one_vs_all = [[column, calculate_h_stat_i_versus(model, X, column, remove_element(X.columns, column))] for
                         column in X.columns]

    return pd.DataFrame(h_stat_one_vs_all,
                        columns=["Feature", "H-statistic"]).sort_values(by="H-statistic",
                                                                        ascending=False,
                                                                        ignore_index=True)


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
        denominator += pd_i_versus ** 2

    return nominator / denominator
