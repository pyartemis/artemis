from typing import List, Dict, Callable

import numpy as np
import pandas as pd
from numpy import ndarray


def remove_element(columns: pd.Index, column) -> List[str]:
    columns_copy = columns.tolist().copy()
    columns_copy.remove(column)

    return columns_copy


def sample_if_not_none(X: pd.DataFrame, n: int):
    if n is None:
        return X
    else:
        return X.sample(n)


def sample_both_if_not_none(X: pd.DataFrame, y: np.array, n: int):
    if n is None:
        return X
    else:
        X_sampled = X.sample(n)
        return X_sampled, y[X_sampled.index]


def all_if_none(X: pd.DataFrame, columns: List[str]):
    if columns is None:
        return X.columns
    else:
        return columns


def center(x: np.array):
    return x - np.mean(x)


def partial_dependence_value(df: pd.DataFrame, change_dict: Dict, predict_function: Callable) -> ndarray:
    assert all(column in df.columns for column in change_dict.keys())
    df_changed = df.assign(**change_dict)
    return np.mean(predict_function(df_changed))
