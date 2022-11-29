from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod


class MinimalDepthImportance(VariableImportanceMethod):

    def __init__(self):
        super().__init__(ImportanceMethod.MINIMAL_DEPTH_IMPORTANCE)

    def importance(self,
                   model,  # to comply with the signature
                   X: Optional[pd.DataFrame] = None,  # to comply with the signature
                   tree_id_to_depth_split: dict = None) -> pd.DataFrame:
        message = f"{self.method} is calculated together with it's interaction counterpart"
        assert tree_id_to_depth_split is not None, message

        columns = _make_column_dict(X)
        feature_to_depth = defaultdict(list)
        for tree_id in tree_id_to_depth_split.keys():
            depth_tree, split_tree = tree_id_to_depth_split[tree_id]
            for f in split_tree.keys():
                feature_to_depth[f].append(np.max(depth_tree) - depth_tree[split_tree[f][0]])

        records_result = []
        for f in feature_to_depth.keys():
            records_result.append({"Feature": columns[f], "Value": np.mean(feature_to_depth[f])})

        self.variable_importance = pd.DataFrame.from_records(records_result).sort_values(
            by="Value", ascending=False, ignore_index=True
        )

        return self.variable_importance


def _make_column_dict(X: pd.DataFrame) -> dict:
    return dict(zip(range(len(X.columns)), X.columns.to_list()))

