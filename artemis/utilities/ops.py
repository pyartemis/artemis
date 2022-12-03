import math
import random
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
        return X, y
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


def partial_dependence_value(df: pd.DataFrame, change_dict: Dict, predict_function: Callable, model) -> ndarray:
    assert all(column in df.columns for column in change_dict.keys())
    df_changed = df.assign(**change_dict)
    return np.mean(predict_function(model, df_changed))


def split_features_num_cat(X, features):
    numerical_cols_set = set(X._get_numeric_data().columns)
    features_set = set(features)

    return features_set.intersection(numerical_cols_set), features_set.difference(numerical_cols_set)
    
    
def point_left_side_circle(x, y, r):
    alpha = math.pi * random.random() + math.pi / 2
    return r * math.cos(alpha) + x, r * math.sin(alpha) + y


def yhat_default(m, d):
    return m.predict(d)


def yhat_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def get_predict_function(model):
    # default extraction
    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return yhat_proba_default
    elif hasattr(model, 'predict'):
        # check if model has predict
        return yhat_default