from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

from artemis.utilities.domain import Method
from artemis.utilities.metrics import Metric, RMSE
from artemis.interactions_methods._method import FeatureInteractionMethod


class ConditionalMinimalDepthMethod(FeatureInteractionMethod):
    def __init__(self, metric: Metric = RMSE()):
        super().__init__(Method.CONDITIONAL_MINIMAL_DEPTH)
        self.metric = metric

    def fit(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        X: pd.DataFrame,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        column_dict = _make_column_dict(X)
        raw_result_df = _calculate_conditional_minimal_depths(model.estimators_, show_progress)
        self.ovo = _summarise_results(raw_result_df, column_dict)


def _make_column_dict(X: pd.DataFrame) -> dict:
    return dict(zip(range(len(X.columns)), X.columns.to_list()))


def _get_node_depths(tree_object: Tree) -> pd.Series:
    current_lvl_ids = [0]
    next_lvl_ids = np.array([])
    depth = np.zeros(tree_object.node_count)
    current_lvl = 0
    while len(current_lvl_ids) > 0:
        depth[current_lvl_ids] = current_lvl
        next_lvl_ids = np.hstack(
            (
                next_lvl_ids,
                tree_object.children_left[current_lvl_ids],
                tree_object.children_right[current_lvl_ids],
            )
        )
        current_lvl_ids = list(next_lvl_ids[next_lvl_ids >= 0].astype(int))
        next_lvl_ids = np.array([])
        current_lvl += 1
    return depth.astype(int)


def _make_tree_df_representation(tree_object: Tree) -> pd.DataFrame:
    tree_df = pd.DataFrame(
        {
            "id": range(tree_object.node_count),
            "left_child": tree_object.children_left,
            "right_child": tree_object.children_right,
            "depth": _get_node_depths(tree_object),
            "split_variable": tree_object.feature,
            "threshold": tree_object.threshold,
        }
    )
    tree_df = (
        tree_df.sort_values(by=["depth", "id"]).loc[tree_df["split_variable"] != -2].reset_index(drop=True)
    )  # rows with -2 are leaves (no split variable)
    tree_df[list(range(tree_object.n_features))] = 0
    return tree_df


def _calculate_conditional_minimal_depth_one_tree(
    decision_tree: Union[DecisionTreeClassifier, DecisionTreeRegressor], id: int
) -> pd.DataFrame:
    tree_object = decision_tree.tree_
    tree_df = _make_tree_df_representation(tree_object)
    offset = np.where(tree_df.columns == 0)[0][0]
    maximal_subtrees_positions = dict(tree_df.groupby("split_variable").depth.idxmin())
    for i in maximal_subtrees_positions.keys():
        start = maximal_subtrees_positions[i]
        df = tree_df.iloc[
            start:,
        ]
        df.iloc[0, i + offset] = 0
        for k in range(1, len(df)):
            k_id = df.iloc[k, 0]
            subset = df.loc[
                (df["left_child"] == k_id) | (df["right_child"] == k_id),
            ]
            if len(subset) != 0 and subset.loc[:, i].values[0] is not None:
                df.iloc[k, i + offset] = (
                    df.loc[(df["left_child"] == k_id) | (df["right_child"] == k_id)][i].values[0] + 1
                )
        tree_df.iloc[start:, i + offset :] = df.iloc[:, i + offset :]
    tree_df.iloc[:, offset:] = tree_df.iloc[:, offset:].replace({0: None})
    tree_result = tree_df.groupby("split_variable").min().iloc[:, offset - 1 :] - 1
    tree_result = tree_result.reset_index()
    tree_result.insert(0, "tree_id", id)
    return tree_result


def _calculate_conditional_minimal_depths(
    trees: List[Union[DecisionTreeClassifier, DecisionTreeRegressor]], show_progress: bool
) -> pd.DataFrame:
    tree_result_list = [
        _calculate_conditional_minimal_depth_one_tree(trees[id], id)
        for id in tqdm(range(len(trees)), disable=not show_progress)
    ]
    forest_result = pd.concat(tree_result_list, ignore_index=True)
    return forest_result


def _summarise_results(raw_result_df: pd.DataFrame, column_dict: dict) -> pd.DataFrame:
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
        .rename({"tree_id": "n_occurences", "value": "mean_min_cond_depth"}, axis=1)
        .sort_values(by=["n_occurences", "mean_min_cond_depth"], ascending=[False, True])
        .reset_index(drop=True)
    )
    final_result["variable"] = final_result["variable"].map(column_dict)
    final_result["root_variable"] = final_result["root_variable"].map(column_dict)
    return final_result
