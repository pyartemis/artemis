from typing import List

import numpy as np
import pandas as pd


def remove_element(columns: pd.Index, column) -> List[str]:
    columns_copy = columns.tolist().copy()
    columns_copy.remove(column)

    return columns_copy


def sample_if_not_none(X: pd.DataFrame, n: int):
    if n is None:
        return X
    else:
        return X.sample(n)


def all_if_none(X: pd.DataFrame, columns: List[str]):
    if columns is None:
        return X.columns
    else:
        return columns


def center(x: np.array):
    return x - np.mean(x)
