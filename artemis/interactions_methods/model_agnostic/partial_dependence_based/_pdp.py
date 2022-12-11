from abc import abstractmethod
from itertools import combinations
from typing import Callable, List, Optional

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
        self.pd_calculator = None
        self.batchsize = None

    @property
    def interactions_ascending_order(self):
        return False

    def fit(self,
            model,
            X: pd.DataFrame,
            n: int = None,
            predict_function: Optional[Callable] = None,
            features: List[str] = None,
            show_progress: bool = False,
            batchsize: int = 2000,
            pd_calculator: Optional[PartialDependenceCalculator] = None):
        """Calculates Partial Dependence Based Feature Interactions Strength and Feature Importance for the given model. 

        Parameters:
        ----------
        model : object
            Model to be explained, should have predict or predict_proba method. 
        X : pd.DataFrame, optional
            Data used to calculate interactions. If n is not None, n rows from X will be sampled. 
        n : int, optional
            Number of samples to be used for calculation of interactions. If None, all rows from X will be used. Default is None.
        predict_function : Callable, optional
            Function used to predict model output. If None, `predict_proba` method will be used if it exists, otherwise `predict` method. Default is None.
        features : List[str], optional
            List of features for which interactions will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        batchsize : int
            Batch size for calculating partial dependence. Prediction requests are collected until the batchsize is exceeded, 
            then the model is queried for predictions jointly for many observations. It speeds up the operation of the method.
            Default is 2000.
        pd_calculator : PartialDependenceCalculator, optional
            PartialDependenceCalculator object containing partial dependence values for a given model and dataset. 
            Providing this object speeds up the calculation as partial dependence values do not need to be recalculated.
            If None, it will be created from scratch. Default is None.
        """
        self.predict_function = get_predict_function(model, predict_function)
        self.model = model
        self.batchsize = batchsize

        self.X_sampled = sample_if_not_none(self._random_generator, X, n)
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

        self._feature_importance_obj = PartialDependenceBasedImportance()
        self.feature_importance = self._feature_importance_obj.importance(self.model, self.X_sampled,
                                                                            features=self.features_included,
                                                                            show_progress=show_progress,
                                                                            pd_calculator=self.pd_calculator)

    @abstractmethod
    def _calculate_ovo_interactions_from_pd(self, show_progress: bool):
        ...
