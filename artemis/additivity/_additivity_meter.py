from typing import List, Optional
import numpy as np
import pandas as pd
from artemis.utilities.domain import ProgressInfoLog
from artemis.utilities.ops import get_predict_function, sample_if_not_none

from artemis.utilities.pd_calculator import PartialDependenceCalculator


class AdditivityMeter:
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.random_generator = np.random.default_rng(random_state)

    def fit(
        self,
        model,
        X: pd.DataFrame,
        n: int = None,
        show_progress: bool = False,
        batchsize: Optional[int] = 2000,
        pd_calculator: Optional[PartialDependenceCalculator] = None,
    ):
        self.predict_function = get_predict_function(model)
        self.model = model
        self.batchsize = batchsize
        self.X_sampled = sample_if_not_none(self.random_generator, X, n)

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

        
