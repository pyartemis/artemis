from collections import defaultdict
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tqdm import tqdm

from artemis.importance_methods.model_specific import MinimalDepthImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import InteractionMethod, VisualisationType
from artemis.utilities.exceptions import MethodNotFittedException


class ConditionalMinimalDepthMethod(FeatureInteractionMethod):
    """
    Class implementing tree-based Conditional Minimal Depth feature interaction method.
    This method is applicable to scikit-learn tree-based models.
    Method is described in the following thesis:
    https://cdn.staticaly.com/gh/geneticsMiNIng/BlackBoxOpener/master/randomForestExplainer_Master_thesis.pdf.

    """

    def __init__(self):
        super().__init__(InteractionMethod.CONDITIONAL_MINIMAL_DEPTH)

    def fit(
            self,
            model: Union[RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier],
            X: pd.DataFrame,
            show_progress: bool = False,
    ):
        """
        Calculate directed one vs one feature interaction profile using Conditional Minimal Depth method.
        Additionally, asses feature importance using average minimal depth importance method.

        Args:
            model:          tree-based model implementing sklearn-like API for which interactions will be extracted
            X:              data used to calculate interactions
            show_progress:  determine whether to show the progress bar
        """
        column_dict = _make_column_dict(X)
        raw_result_df, trees = _calculate_conditional_minimal_depths(model.estimators_, len(X.columns), show_progress)
        self.ovo = _summarise_results(raw_result_df, column_dict, self.method)
        self.variable_importance = MinimalDepthImportance().importance(model, X, trees)

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        """See `plot` documentation in `FeatureInteractionMethod`."""
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualisation.plot(self.ovo,
                                vis_type,
                                feature_column_name_1="root_variable",
                                feature_column_name_2="variable",
                                directed=True,
                                variable_importance=self.variable_importance)


def _calculate_conditional_minimal_depths(
        trees: List[Union[DecisionTreeClassifier, DecisionTreeRegressor]], column_size: int, show_progress: bool
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate conditional minimal depth for all `trees`.
    For further use in variable importance, it also returns dictionary of depths and split variables
    for each tree in `trees`.

    Args:
        trees:          list of decision trees
        column_size:    number of features in the data
        show_progress:  determine whether to show the progress bar

    Returns:
        Conditional minimal depths, depths and split variables of all trees

    """
    tree_result_list = list()
    tree_id_to_depth_split = dict()
    for tree_id in tqdm(range(len(trees)), disable=not show_progress):
        tree_repr = _tree_representation(trees[tree_id].tree_)
        depths, split_variable_to_node_id = _bfs(tree_repr)
        tree_id_to_depth_split[tree_id] = [depths, split_variable_to_node_id]
        tree_result_list.append(_conditional_minimal_depth(split_variable_to_node_id, depths, tree_id, column_size))

    forest_result = pd.concat(tree_result_list, ignore_index=True)
    return forest_result, tree_id_to_depth_split


def _bfs(tree: pd.DataFrame):
    """
    Run BFS algorithm for a given `tree`.
    Calculates depth of each node and creates a map of split_variable to nodes using this split_variable.
    This representation is useful for efficient conditional minimal depth calculation.

    Args:
        tree: tree representation
    Returns:
        depth of each node, nodes grouped by split_variable
    """
    current_lvl_ids = [0]
    depth = np.zeros(len(tree)).astype(int)
    current_depth = 0
    split_variable_to_node_id = defaultdict(lambda: np.ndarray(0).astype(int))

    while len(current_lvl_ids) > 0:
        depth[current_lvl_ids] = current_depth
        grouped_by_split_var = _split_var_to_ids(current_lvl_ids, tree)
        _append_nodes_to_split_var(grouped_by_split_var, split_variable_to_node_id)

        current_lvl_ids = _next_level(current_lvl_ids, tree)
        current_depth += 1

    _delete_leaves(split_variable_to_node_id)

    return depth, split_variable_to_node_id


def _tree_representation(decision_tree_object):
    """Efficient, sklearn-like table tree representation. Row = node in the tree."""
    return pd.DataFrame({
        "id": range(decision_tree_object.node_count),
        "left_child": decision_tree_object.children_left,
        "right_child": decision_tree_object.children_right,
        "split_variable": decision_tree_object.feature,
    })


def _conditional_minimal_depth(split_var_to_nodes: dict, depths: np.array, tree_id: int, column_size: int):
    """
    Main algorithm for efficient conditional minimal depth calculation.
    Intuition: for each pair of features (f_1, f_2), find lowest-depth nodes (N) splitting f_1 and calculate distance to
    the closest node in N subtree, using f_2 as a split variable. f_1 -> f_2 conditional minimal distance is minimum
    over all such distances.

    Specifics can be found in:
    https://cdn.staticaly.com/gh/geneticsMiNIng/BlackBoxOpener/master/randomForestExplainer_Master_thesis.pdf
    """

    conditional_depths = []

    n_nodes = len(depths)
    for f_1 in range(column_size):
        for f_2 in range(column_size):

            min_val = float('+inf')
            if f_1 in split_var_to_nodes and f_2 in split_var_to_nodes:
                visited = np.full(n_nodes, False)
                split_f1, split_f2 = split_var_to_nodes[f_1], split_var_to_nodes[f_2]
                for i in range(len(split_f1)):

                    lower_bound = current_root_id = split_f1[i]
                    if visited[current_root_id]:
                        continue

                    current_root_depth = depths[current_root_id]
                    upper_bound = _calculate_maximal_id_in_subtree(current_root_depth, current_root_id, depths, n_nodes)

                    visited[lower_bound:upper_bound] = True

                    id_f2_in_subtree = _f_2_split_nodes_in_f_1_subtree(current_root_id, split_f2, upper_bound)
                    if len(id_f2_in_subtree) > 0:
                        min_val = min(min_val, np.min(depths[split_f2[id_f2_in_subtree]]) - current_root_depth - 1)

            conditional_depths.append({"split_variable": f_1, "ancestor_variable": f_2, "value": min_val})

    res = pd.DataFrame.from_records(conditional_depths).replace({float("+inf"): None})

    return _map_to_summarise_format(res, tree_id)


def _make_column_dict(X: pd.DataFrame) -> dict:
    return dict(zip(range(len(X.columns)), X.columns.to_list()))


def _summarise_results(raw_result_df: pd.DataFrame, column_dict: dict, method_name: str) -> pd.DataFrame:
    """Average result over trees, rename columns, fill missing data."""
    final_result = pd.melt(
        raw_result_df, id_vars=["tree_id", "split_variable"], value_vars=list(range(len(column_dict)))
    )
    final_result = (
        final_result.rename({"split_variable": "variable", "variable": "root_variable"}, axis=1)[
            ~pd.isna(final_result["value"])
        ]
        .groupby(["variable", "root_variable"])
        .agg({"tree_id": "size", "value": "mean"})
        .reset_index()
        .rename({"tree_id": "n_occurences", "value": method_name}, axis=1)
        .sort_values(by=["n_occurences", method_name], ascending=[False, True])
        .reset_index(drop=True)
    )
    final_result["variable"] = final_result["variable"].map(column_dict)
    final_result["root_variable"] = final_result["root_variable"].map(column_dict)

    return final_result[final_result["variable"] != final_result["root_variable"]]


def _delete_leaves(split_variable_to_node_id: dict):
    """Leaves are indicated by -2, and have no split variable."""
    del split_variable_to_node_id[-2]


def _calculate_maximal_id_in_subtree(current_root_depth: int, current_root_id: int, depths: np.array, n_nodes: int):

    """Determine the greatest id of the node in a subtree rooted in `current_root_id` """
    upper_bound_set = np.where((depths <= current_root_depth) & (np.arange(n_nodes) > current_root_id))[0]
    upper_bound = n_nodes

    if len(upper_bound_set) > 0:
        upper_bound = np.min(upper_bound_set)

    return upper_bound


def _next_level(current_lvl_ids: list, tree: pd.DataFrame):
    """BFS one step, add `current_lvl_ids` ancestors to process."""
    current_lvl_ids = np.hstack((tree.loc[current_lvl_ids, "left_child"], tree.loc[current_lvl_ids, "right_child"]))
    current_lvl_ids = list(current_lvl_ids[current_lvl_ids >= 0].astype(int))  # -1 denotes termination node

    return current_lvl_ids


def _append_nodes_to_split_var(grouped_by_split_var: pd.DataFrame, split_variable_to_node_id: dict):
    for split_var, ids in grouped_by_split_var.items():
        split_variable_to_node_id[split_var] = np.hstack((split_variable_to_node_id[split_var], ids))


def _split_var_to_ids(current_lvl_ids: list, tree: pd.DataFrame):
    return tree.loc[current_lvl_ids, ["id", "split_variable"]].groupby("split_variable")["id"].apply(np.array)


def _f_2_split_nodes_in_f_1_subtree(current_root_id: int, split_f2: np.array, upper_bound: int):
    return np.where((split_f2 > current_root_id) & (split_f2 < upper_bound))[0]


def _map_to_summarise_format(res: pd.DataFrame, tree_id: int):
    tree_result = res.pivot_table("value", "ancestor_variable", "split_variable").rename_axis(None).rename_axis(None,
                                                                                                                axis=1)
    tree_result.insert(0, "tree_id", tree_id)
    tree_result.insert(0, "split_variable", range(len(tree_result)))

    return tree_result
