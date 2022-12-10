from abc import abstractmethod
from itertools import combinations
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import get_predict_function, sample_if_not_none, all_if_none
from artemis.utilities.pd_calculator import PartialDependenceCalculator


class PartialDependenceBasedMethod(FeatureInteractionMethod):

    def __init__(self, method: str, random_state: Optional[int] = None):
        super().__init__(method, random_state=random_state)

    @property
    def interactions_ascending_order(self):
        return False

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            features: List[str] = None,
            show_progress: bool = False,
            batchsize: Optional[int] = 2000,
            pd_calculator: Optional[PartialDependenceCalculator] = None):
        """Calculates Partial Dependence Based Interactions and Importance for the given model. 

        Parameters:
            model -- model to be explained
            X (pd.DataFrame, optional) -- data used to calculate interactions
            n (int, optional) -- number of samples to be used for calculation of interactions
            features (List[str], optional) -- list of features for which interactions will be calculated
            show_progress (bool) -- whether to show progress bar 
        """
        self.predict_function = get_predict_function(model)
        self.model = model
        self.batchsize = batchsize

        self.X_sampled = sample_if_not_none(self.random_generator, X, n)
        self.features_included = all_if_none(X.columns, features)
        self.pairs = list(combinations(self.features_included, 2))

        if pd_calculator is None:
            self.pd_calculator = PartialDependenceCalculator(self.model, self.X_sampled, self.predict_function, self.batchsize)
        else: 
            if pd_calculator.model != self.model:
                raise ValueError("Model in PDP calculator is different than the model in the method.")
            if not pd_calculator.X.equals(self.X_sampled):
                raise ValueError("Data in PDP calculator is different than the data in the method.")
            self.pd_calculator = pd_calculator

        self.ovo = self._calculate_ovo_interactions_from_pd(show_progress = show_progress)

        self._variable_importance_obj = PartialDependenceBasedImportance()
        self.variable_importance = self._variable_importance_obj.importance(self.model, self.X_sampled,
                                                                            features=self.features_included,
                                                                            show_progress=show_progress,
                                                                            pd_calculator=self.pd_calculator)

    @abstractmethod
    def _calculate_ovo_interactions_from_pd(self, show_progress: bool):
        ...
