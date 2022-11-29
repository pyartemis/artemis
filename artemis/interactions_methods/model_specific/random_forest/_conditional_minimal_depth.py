from collections import defaultdict
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tqdm import tqdm

from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.domain import Method, VisualisationType


class ConditionalMinimalDepthMethod(FeatureInteractionMethod):
    def __init__(self):
        super().__init__(Method.CONDITIONAL_MINIMAL_DEPTH)

    def fit(
            self,
            model: Union[RandomForestClassifier, RandomForestRegressor],
            X: pd.DataFrame,
            show_progress: bool = False,
    ):
        column_dict = _make_column_dict(X)
        raw_result_df = _calculate_conditional_minimal_depths(model.estimators_, len(X.columns), show_progress)
        self.ovo = _summarise_results(raw_result_df, column_dict, self.method)

    def plot(self, vis_type: str = VisualisationType.SUMMARY):
        assert self.ovo is not None, "Before executing plot() method, fit() must be executed!"
        self.visualisation.plot(self.ovo,
                                vis_type,
                                feature_column_name_1="root_variable",
                                feature_column_name_2="variable",
                                directed=True)


def _calculate_conditional_minimal_depths(
        trees: List[Union[DecisionTreeClassifier, DecisionTreeRegressor]], column_size: int, show_progress: bool
) -> pd.DataFrame:
    tree_result_list = list()
    for tree_id in tqdm(range(len(trees)), disable=not show_progress):
        tree_repr = _tree_representation(trees[tree_id].tree_)
        depths, split_variable_to_node_id = bfs(tree_repr)
        tree_result_list.append(_conditional_minimal_depth(split_variable_to_node_id, depths, tree_id, column_size))

    forest_result = pd.concat(tree_result_list, ignore_index=True)
    return forest_result


def bfs(tree):
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
    return pd.DataFrame({
        "id": range(decision_tree_object.node_count),
        "left_child": decision_tree_object.children_left,
        "right_child": decision_tree_object.children_right,
        "split_variable": decision_tree_object.feature,
    })


def _conditional_minimal_depth(split_var_to_nodes: dict, depths: np.array, tree_id: int, column_size: int):
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
    del split_variable_to_node_id[-2]


def _calculate_maximal_id_in_subtree(current_root_depth: int, current_root_id: int, depths: np.array, n_nodes: int):
    upper_bound_set = np.where((depths <= current_root_depth) & (np.arange(n_nodes) > current_root_id))[0]
    upper_bound = n_nodes

    if len(upper_bound_set) > 0:
        upper_bound = np.min(upper_bound_set)

    return upper_bound


def _next_level(current_lvl_ids: list, tree: pd.DataFrame):
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
