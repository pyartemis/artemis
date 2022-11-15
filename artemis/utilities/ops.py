import math
import random
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


def point_on_circle(x, y, r):
    alpha = 2 * math.pi * random.random()

    return r * math.cos(alpha) + x, r * math.sin(alpha) + y
