from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import FeatureImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog
from artemis.utilities.ops import (
    all_if_none,
    get_predict_function,
    sample_if_not_none,
    split_features_num_cat,
)
from artemis.utilities.pd_calculator import PartialDependenceCalculator


class PartialDependenceBasedImportance(FeatureImportanceMethod):
    """
    Partial Dependence Based Feature Importance.
    It is used for calculating feature importance for partial dependence based feature interaction methods:
    Friedman's H-statistic and Greenwell methods.


    Attributes:
    ----------
    method : str 
        Method name.
    feature_importance : pd.DataFrame 
        Feature importance values.
    features_included : List[str]
        List of features for which importance is calculated.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    pd_calculator : PartialDependenceCalculator
        Object used to calculate and store partial dependence values.

    References:
    ----------
    - https://arxiv.org/abs/1805.04755
    """

    def __init__(self):
        """Constructor for PartialDependenceBasedImportance"""
        super().__init__(ImportanceMethod.PDP_BASED_IMPORTANCE)

    def importance(
        self,
        model,
        X: pd.DataFrame,
        n: int = None,
        predict_function: Optional[Callable] = None,
        features: Optional[List[str]] = None,
        show_progress: bool = False,
        batchsize: int = 2000,
        pd_calculator: Optional[PartialDependenceCalculator] = None,
    ):
        """Calculates Partial Dependence Based Feature Importance.
        
        Parameters:
        ----------
        model : object
             Model for which importance will be calculated, should have predict_proba or predict method, or predict_function should be provided. 
        X : pd.DataFrame
             Data used to calculate importance. If n is not None, n rows from X will be sampled. 
        n : int, optional
            Number of samples to be used for calculation of importance. If None, all rows from X will be used. Default is None.
        predict_function : Callable, optional
            Function used to predict model output. It should take model and dataset and outputs predictions. 
            If None, `predict_proba` method will be used if it exists, otherwise `predict` method. Default is None.
        features : List[str], optional
            List of features for which importance will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        batchsize : int
            Batch size for calculating partial dependence. Data for prediction are collected until the number of rows exceeds batchsize. 
            Then, the `predict_function` is called, jointly for the entire batch of observations. It speeds up the operation of the method
            by reducing the number of `predict_function` calls.
            Default is 2000.
        pd_calculator : PartialDependenceCalculator, optional
            PartialDependenceCalculator object containing partial dependence values for a given model and dataset. 
            Providing this object speeds up the calculation as partial dependence values do not need to be recalculated.
            If None, it will be created from scratch. Default is None.

        Returns:
        -------
        pd.DataFrame
            Result dataframe containing feature importance with columns: "Feature", "Importance"
        """
        self.predict_function = get_predict_function(model, predict_function)
        self.X_sampled = sample_if_not_none(self._random_generator, X, n)
        self.features_included = all_if_none(X.columns, features)
  

        if pd_calculator is None:
            self.pd_calculator = PartialDependenceCalculator(model, self.X_sampled, self.predict_function, batchsize)
        else: 
            if pd_calculator.model != model:
                raise ValueError("Model in PDP calculator is different than the model in the method.")
            if not pd_calculator.X.equals(self.X_sampled):
                raise ValueError("Data in PDP calculator is different than the data in the method.")
            self.pd_calculator = pd_calculator

        self.feature_importance = self._pdp_importance(show_progress)
        return self.feature_importance

    @property
    def importance_ascending_order(self):
        return False

    def _pdp_importance(self, show_progress: bool) -> pd.DataFrame:
        self.pd_calculator.calculate_pd_single(show_progress=show_progress)

        importance = []
        num_features, _ = split_features_num_cat(self.X_sampled, self.features_included)

        for feature in self.features_included:
            pdp = self.pd_calculator.get_pd_single(feature)
            importance.append(_calc_importance(feature, pdp, feature in num_features))

        return pd.DataFrame(importance, columns=["Feature", "Importance"]).sort_values(
            by="Importance", ascending=self.importance_ascending_order, ignore_index=True
        ).fillna(0)


def _calc_importance(feature: str, pdp: np.ndarray, is_numerical: bool):
    return [feature, np.std(pdp) if is_numerical else (np.max(pdp) - np.min(pdp)) / 4]
    
