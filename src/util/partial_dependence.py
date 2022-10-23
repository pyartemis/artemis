from typing import Dict, Callable
import numpy as np
import pandas as pd
from numpy import ndarray


def partial_dependence_value(df: pd.DataFrame, change_dict: Dict, predict_function: Callable) -> ndarray:
    assert all(column in df.columns for column in change_dict.keys())
    df_changed = df.assign(**change_dict)
    return np.mean(predict_function(df_changed))
