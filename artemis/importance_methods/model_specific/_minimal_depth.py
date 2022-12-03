from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod, InteractionMethod
from artemis.utilities.exceptions import FeatureImportanceWithoutInteractionException


class MinimalDepthImportance(VariableImportanceMethod):
    """
    Class implementing Minimal Depth Feature Importance. It applies to tree-based models.
    It uses data calculated in Conditional Minimal Depth method and so needs to be calculated together.

    Importance of a feature in the forest is defined as average distance to terminal nodes from the highest node using
    this feature as a split variable.
    """

    def __init__(self):
        super().__init__(ImportanceMethod.MINIMAL_DEPTH_IMPORTANCE)

    def importance(self,
                   model,  # to comply with the signature
                   X: Optional[pd.DataFrame] = None,
                   tree_id_to_depth_split: dict = None) -> pd.DataFrame:
        """
        Calculate minimal depth feature importance.

        Args:
            model:                      unused, comply with signature
            X:                          data to use for calculating importance on
            tree_id_to_depth_split:     precalculated in ConditionalMinimalDepth depths and split_variables map

        Returns:
            minimal depth feature importance
        """
        _check_preconditions(self.method, tree_id_to_depth_split)

        columns = _make_column_dict(X)
        feature_to_depth = defaultdict(list)
        for tree_id in tree_id_to_depth_split.keys():
            depth_tree, split_tree = tree_id_to_depth_split[tree_id]
            for f in split_tree.keys():
                # until we do not handle this in visualisations, let's keep higher = bigger axiom
                feature_to_depth[f].append(np.max(depth_tree) - depth_tree[split_tree[f][0]])

        records_result = []
        for f in feature_to_depth.keys():
            records_result.append({"Feature": columns[f], "Value": np.mean(feature_to_depth[f])})

        self.variable_importance = pd.DataFrame.from_records(records_result).sort_values(
            by="Value", ignore_index=True
        )

        return self.variable_importance


def _check_preconditions(method: str, tree_id_to_depth_split: dict):
    if tree_id_to_depth_split is None:
        raise FeatureImportanceWithoutInteractionException(method, InteractionMethod.CONDITIONAL_MINIMAL_DEPTH)


def _make_column_dict(X: pd.DataFrame) -> dict:
    return dict(zip(range(len(X.columns)), X.columns.to_list()))
