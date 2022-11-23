from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, ProgressInfoLog
from artemis.utilities.ops import partial_dependence_value, split_features_num_cat


class PartialDependenceBasedImportance(VariableImportanceMethod):

    def __init__(self):
        super().__init__(ImportanceMethod.PDP_BASED_IMPORTANCE)

    def importance(self,
                   model,
                   X: pd.DataFrame,
                   features: List[str] = None,
                   show_progress: bool = False,
                   precalculated_pdp: dict = None):

        if precalculated_pdp is None:
            self.variable_importance = _pdp_importance(model, X, features, show_progress, self.method)
        else:
            self.variable_importance = _map_to_df(X, features, precalculated_pdp, self.method)

        return self.variable_importance


def _map_to_df(X, features, precalculated_pdp: dict, method):
    importance = list()
    num_features, cat_features = split_features_num_cat(X, features)

    for f in precalculated_pdp.keys():
        importance.append(_calc_importance(f, method, precalculated_pdp[f], f in num_features))

    return pd.DataFrame.from_records(importance).sort_values(
        by=method, ascending=False, ignore_index=True
    )


def _pdp_importance(model, X, features, progress, method) -> pd.DataFrame:
    importance = []

    num_features, cat_features = split_features_num_cat(X, features)

    for feature in tqdm(features, disable=not progress, desc=ProgressInfoLog.CALC_VAR_IMP):
        pdp = list()
        for _, row in X.iterrows():
            pdp.append(partial_dependence_value(X, {feature: row[feature]}, model.predict))

        importance.append(_calc_importance(feature, method, pdp, feature in num_features))

    return pd.DataFrame.from_records(importance).sort_values(
        by=method, ascending=False, ignore_index=True
    )


def _calc_importance(feature, method, pdp, is_numerical):
    return {"Feature": feature, method: np.std(pdp) if is_numerical else (np.max(pdp) - np.min(pdp)) / 4}
