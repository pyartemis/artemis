from abc import abstractmethod

import pandas as pd

from artemis.utilities.domain import VisualisationType
from artemis.visualisation.configuration import VisualisationConfigurationProvider
from artemis.visualisation.visualisation import Visualisation


class VariableImportanceMethod:

    def __init__(self, method: str):
        self.method = method

    @abstractmethod
    def importance(self, model, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
