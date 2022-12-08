from abc import abstractmethod

import pandas as pd

from artemis.utilities.domain import ImportanceOrderProvider


class VariableImportanceMethod:

    def __init__(self, method: str):
        self.method = method
        self.variable_importance = None
        self.importance_ascending_order = ImportanceOrderProvider.is_ascending_order(method)

    @abstractmethod
    def importance(self, model, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
