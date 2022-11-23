from abc import abstractmethod

import pandas as pd


class VariableImportanceMethod:

    def __init__(self, method: str):
        self.method = method
        self.variable_importance = None

    @abstractmethod
    def importance(self, model, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
