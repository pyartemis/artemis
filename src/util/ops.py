from typing import List

import pandas as pd
from pandas import Index


def remove_element(columns: Index, column) -> List[str]:
    columns_copy = columns.tolist().copy()
    columns_copy.remove(column)

    return columns_copy


def sample_if_not_none(X: pd.DataFrame, n: int):
    if n is None:
        return X
    else:
        return X.sample(n)
