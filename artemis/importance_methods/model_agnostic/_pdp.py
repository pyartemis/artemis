from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog
from artemis.utilities.ops import (
    get_predict_function,
    partial_dependence_value,
    split_features_num_cat,
)


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
        features: Optional[List[str]] = None,
        show_progress: bool = False,
        precalculated_pdp: dict = None,
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

        # if precalculated_pdp is None:
        self.variable_importance = _pdp_importance(
            self.predict_function, self.model, X, features, show_progress
        )
        # else:
        #    self.variable_importance = _map_to_df(X, features, precalculated_pdp)

        return self.variable_importance


def _map_to_df(X: pd.DataFrame, features: List[str], precalculated_pdp: dict):
    importance = list()
    num_features, cat_features = split_features_num_cat(X, features)

    feature_to_pdp = defaultdict(list)
    for feature, val in precalculated_pdp.keys():
        feature_to_pdp[feature].append(val)

    for feature in feature_to_pdp.keys():
        importance.append(
            _calc_importance(feature, feature_to_pdp[feature], feature in num_features)
        )

    return pd.DataFrame.from_records(importance).sort_values(
        by="Value", ascending=False, ignore_index=True
    )


def _pdp_importance(
    predict_function, model, X: pd.DataFrame, features: List[str], progress: bool
) -> pd.DataFrame:
    importance = []

    num_features, cat_features = split_features_num_cat(X, features)

    for feature in tqdm(
        features, disable=not progress, desc=ProgressInfoLog.CALC_VAR_IMP
    ):
        pdp = list()
        for _, row in X.iterrows():
            pdp.append(
                partial_dependence_value(
                    X, {feature: row[feature]}, predict_function, model
                )
            )

        importance.append(_calc_importance(feature, pdp, feature in num_features))

    return pd.DataFrame.from_records(importance).sort_values(
        by="Value", ascending=False, ignore_index=True
    )


def _calc_importance(feature: str, pdp: List, is_numerical: bool):
    return {
        "Feature": feature,
        "Value": np.std(pdp) if is_numerical else (np.max(pdp) - np.min(pdp)) / 4,
    }
