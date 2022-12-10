from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog
from artemis.utilities.ops import (
    all_if_none,
    get_predict_function,
    partial_dependence_value,
    sample_if_not_none,
    split_features_num_cat,
)
from artemis.utilities.pd_calculator import PartialDependenceCalculator


class PartialDependenceBasedImportance(VariableImportanceMethod):
    """Class implementing Partial Dependence Based Feature Importance.
    It is used for calculating feature importance for partial dependence based feature interaction methods:
    Friedman H-statistic and Greenwell methods.

    References:
    - https://arxiv.org/abs/1805.04755
    """

    def __init__(self):
        super().__init__(ImportanceMethod.PDP_BASED_IMPORTANCE)

    def importance(
        self,
        model,
        X: pd.DataFrame,
        n: int = None,
        features: Optional[List[str]] = None,
        show_progress: bool = False,
        batchsize: Optional[int] = 2000,
        pd_calculator: Optional[PartialDependenceCalculator] = None,
    ):
        """Calculate Partial Dependence Based Feature Importance.
        Arguments:
            model -- model for which importance will be extracted
            X (pd.DataFrame) -- data used to calculate importance
            features (List[str], optional) -- list of features that will be used during importance calculation
            show_progress (bool) -- determine whether to show the progress bar
            precalculated_pdp (dict) --  precalculated partial dependence profiles, if None calculated from scratch

        Returns:
            pd.DataFrame -- DataFrame containing feature importance with columns: "Feature", "Importance"
        """
        self.predict_function = get_predict_function(model)
        self.model = model
        self.batchsize = batchsize

        self.X_sampled = sample_if_not_none(self.random_generator, X, n)
        self.features_included = all_if_none(X.columns, features)
  

        if pd_calculator is None:
            self.pd_calculator = PartialDependenceCalculator(self.model, self.X_sampled, self.predict_function, self.batchsize)
        else: 
            if pd_calculator.model != self.model:
                raise ValueError("Model in PDP calculator is different than the model in the method.")
            if not pd_calculator.X.equals(self.X_sampled):
                raise ValueError("Data in PDP calculator is different than the data in the method.")
            self.pd_calculator = pd_calculator

        self.variable_importance = self._pdp_importance(show_progress)
        return self.variable_importance

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
    
