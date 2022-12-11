from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import get_predict_function, sample_if_not_none

from artemis.utilities.pd_calculator import PartialDependenceCalculator


class AdditivityMeter:
    """
    AdditivityMeter is a class that calculates the additivity index of a model.

    Parameters:
    -----------
    additivity_index : float
        Additivity index of the model.
    full_result : pd.DataFrame
        Dataframe with the results of the additivity index calculation. 
        It contains centered partial dependence values and prediction for every observation and feature.
    preds: np.ndarray
        Predictions for the sampled data.
    model : object
        Model for which additivity index is calculated.
    X_sampled: pd.DataFrame
        Sampled data used for calculation.
    pd_calculator : PartialDependenceCalculator
        Object used to calculate and store partial dependence values.
    batchsize: int
        Batch size used for calculation.
    
    """
    def __init__(self, random_state: Optional[int] = None):
        self._random_generator = np.random.default_rng(random_state)

    def fit(
        self,
        model,
        X: pd.DataFrame,
        n: int = None,
        predict_function: Optional[Callable] = None,
        show_progress: bool = False,
        batchsize: int = 2000,
        pd_calculator: Optional[PartialDependenceCalculator] = None,
    ):
        """ 
        Calculates the additivity index of the given model.

        Parameters:
        -----------
        model : object
            Model to be explained, should have predict_proba or predict method, or predict_function should be provided. 
        X : pd.DataFrame
            Data used to calculate the additivity index. If n is not None, n rows from X will be sampled. 
        n : int, optional
            Number of samples to be used for calculation of the additivity index. If None, all rows from X will be used. Default is None.
        predict_function : Callable, optional
            Function used to predict model output. It should take model and dataset and outputs predictions. 
            If None, `predict_proba` method will be used if it exists, otherwise `predict` method. Default is None.
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

        Returns:
        --------
        additivity_index : float
            Additivity index of the model. Value from [0, 1] interval where 1 means that the model is additive, 
            and 0 means that the model is not additive.
        """
        self.predict_function = get_predict_function(model, predict_function)
        self.model = model
        self.batchsize = batchsize
        self.X_sampled = sample_if_not_none(self._random_generator, X, n)

        if pd_calculator is None:
            self.pd_calculator = PartialDependenceCalculator(
                self.model, self.X_sampled, self.predict_function, self.batchsize
            )
        else:
            if pd_calculator.model != self.model:
                raise ValueError(
                    "Model in PDP calculator is different than the model in the method."
                )
            if not pd_calculator.X.equals(self.X_sampled):
                raise ValueError(
                    "Data in PDP calculator is different than the data in the method."
                )
            self.pd_calculator = pd_calculator

        self.full_result = self.X_sampled.copy()
        self.additivity_index = self._calculate_additivity(show_progress=show_progress)
        return self.additivity_index

    def _calculate_additivity(self, show_progress: bool):
        self.pd_calculator.calculate_pd_single(
            show_progress=show_progress, desc=ProgressInfoLog.CALC_ADD
        )

        self.preds = self.predict_function(self.model, self.X_sampled)
        for var in self.X_sampled.columns:
            self.full_result[var] = self.pd_calculator.get_pd_single(var, self.X_sampled[var].values) - np.mean(self.preds)
        
        self.full_result = self.full_result 
        self.full_result["centered_prediction"] = self.preds - np.mean(self.preds)

        sum_first_order_effects = self.full_result.values[:, :-1].sum(axis=1) + np.mean(self.preds)
        return 1-np.sum((self.preds - sum_first_order_effects)**2) / np.sum((self.full_result["centered_prediction"])**2)

        
