from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from artemis.importance_methods._method import FeatureImportanceMethod
from artemis.utilities.domain import ImportanceMethod, InteractionMethod
from artemis.utilities.exceptions import FeatureImportanceWithoutInteractionException


class MinimalDepthImportance(FeatureImportanceMethod):
    """
    Minimal Depth Feature Importance.
    It applies to tree-based models like Random Forests.
    It uses data calculated in ConditionalMinimalDepth method from `interactions_methods` module and so needs to be calculated together.

    Importance of a feature is defined as the lowest depth of node using this feature as a split variable in a tree, averaged over all trees.

    Attributes:
    ----------
    method : str 
        Method name.
    feature_importance : pd.DataFrame 
        Feature importance values.
        
    References:
    ----------
    - https://modeloriented.github.io/randomForestExplainer/
    - https://doi.org/10.1198/jasa.2009.tm08622
    """

    def __init__(self):
        """Constructor for MinimalDepthImportance"""
        super().__init__(ImportanceMethod.MINIMAL_DEPTH_IMPORTANCE)

    def importance(
        self,
        model,  
        tree_id_to_depth_split: dict,
    ) -> pd.DataFrame:
        """Calculates Minimal Depth Feature Importance.

        Parameters:
        ----------
        model : object
               Model for which importance will be calculated, should have predict method.
        tree_id_to_depth_split : dict
            Dictionary containing minimal depth of each node in each tree.

        Returns:
        -------
        pd.DataFrame
            Result dataframe containing feature importance with columns: "Feature", "Importance"
        """
        _check_preconditions(self.method, tree_id_to_depth_split)

        columns = _make_column_dict(model.feature_names_in_)
        feature_to_depth = defaultdict(list)
        for tree_id in tree_id_to_depth_split.keys():
            depth_tree, split_tree = tree_id_to_depth_split[tree_id]
            for f in split_tree.keys():
                feature_to_depth[f].append(depth_tree[split_tree[f][0]])


        records_result = []
        for f in feature_to_depth.keys():
            records_result.append(
                {"Feature": columns[f], "Importance": np.mean(feature_to_depth[f])}
            )

        self.feature_importance = pd.DataFrame.from_records(
            records_result
        ).sort_values(by="Importance", ignore_index=True)

        return self.feature_importance

    @property
    def importance_ascending_order(self):
        return True



def _check_preconditions(method: str, tree_id_to_depth_split: dict):
    if tree_id_to_depth_split is None:
        raise FeatureImportanceWithoutInteractionException(
            method, InteractionMethod.CONDITIONAL_MINIMAL_DEPTH
        )


def _make_column_dict(columns: np.ndarray) -> dict:
    return dict(zip(range(len(columns)), list(columns)))
