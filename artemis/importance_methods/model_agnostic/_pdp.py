from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod
from artemis.utilities.ops import partial_dependence_value


class PartialDependenceBasedImportance(VariableImportanceMethod):

    def __init__(self):
        super().__init__(ImportanceMethod.PDP_BASED_IMPORTANCE)

    def fit(self,
            model,
            X: pd.DataFrame,
            features: List[str] = None,
            show_progress: bool = False,
            precalculated_pdp: dict = None):

        if precalculated_pdp is None:
            self.variable_importance = _pdp_importance(model, X, features, show_progress, self.method)
        else:
            self.variable_importance = _map_to_df(precalculated_pdp, self.method)


def _map_to_df(precalculated_pdp: dict, method):
    importance = list()
    for f in precalculated_pdp.keys():
        importance.append({"Feature": f, method: np.std(precalculated_pdp[f])})

    return pd.DataFrame.from_records(importance).sort_values(
        by=method, ascending=False, ignore_index=True
    )


def _pdp_importance(model, X, features, progress, method) -> pd.DataFrame:
    importance = []
    for feature in tqdm(features, disable=not progress):
        pdp = list()
        for _, row in X.iterrows():
            pdp.append(partial_dependence_value(X, {feature: row[feature]}, model.predict))

        # TODO: change to (max - min)/4 for categorical feature
        importance.append({"Feature": feature, method: np.std(pdp)})

    return pd.DataFrame.from_records(importance).sort_values(
        by=method, ascending=False, ignore_index=True
    )
