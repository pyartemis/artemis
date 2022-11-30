from artemis.importance_methods._method import VariableImportanceMethod
from artemis.utilities.domain import ImportanceMethod
from artemis.interactions_methods.model_specific.gb_trees._handler import GBTreesHandler
from artemis.utilities.split_score_metrics import SplitScoreImportanceMetric

import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm


class SplitScoreImportance(VariableImportanceMethod):
    def __init__(self):
        super().__init__(ImportanceMethod.SPLIT_SCORE_IMPORTANCE)

    def importance(
        self,
        model,
        X: Optional[pd.DataFrame] = None,  # unused as explanations are calculated only for trained model, left for compatibility
        features: Optional[List[str]] = None,
        selected_metric: str = SplitScoreImportanceMetric.SUM_GAIN,
        show_progress: bool = False,
        trees_df: Optional[pd.DataFrame] = None,
    ):

        if trees_df is None:
            if not isinstance(model, GBTreesHandler):
                model = GBTreesHandler(model)
            trees_df = model.trees_df

        if trees_df["depth"].isnull().values.any():
            trees_df = _calculate_depth(trees_df, show_progress)
        self.full_result = _calculate_all_variable_importance(
            trees_df, features, selected_metric
        )
        self.variable_importance = _select_metric(self.full_result, selected_metric)
        return self.variable_importance


def _calculate_all_variable_importance(
    trees_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    selected_metric: str = SplitScoreImportanceMetric.SUM_GAIN,
):
    if features is not None:
        trees_df = trees_df.loc[trees_df["split_feature"].isin(features)]
    else:
        trees_df = trees_df.loc[trees_df["split_feature"] != "Leaf"]

    basic_metrics = _calculate_basic_metrics(trees_df)
    root_metrics = _calculate_root_metrics(trees_df)
    mean_weighted_depth = _calculate_mean_weighted_depth_metric(trees_df)

    importance_full_result = basic_metrics.join(root_metrics)
    importance_full_result = pd.concat(
        [importance_full_result, mean_weighted_depth], axis=1
    ).reset_index()

    return importance_full_result.sort_values(selected_metric, ascending=False)


def _calculate_basic_metrics(trees_df: pd.DataFrame):
    importance_full_result = trees_df.groupby("split_feature").agg(
        mean_gain=("gain", "mean"),
        sum_gain=("gain", "sum"),
        mean_cover=("cover", "mean"),
        sum_cover=("cover", "sum"),
        mean_depth=("depth", "mean"),
    )
    return importance_full_result


def _calculate_root_metrics(trees_df: pd.DataFrame):
    root_freq_df = (
        trees_df.loc[trees_df["depth"] == 0]
        .groupby("split_feature")
        .agg(root_frequency=("tree", "count"), sum_gain_root=("gain", "sum"))
    )
    sum_gain = np.sum(root_freq_df.sum_gain_root)
    root_freq_df["weighted_root_frequency"] = (
        root_freq_df["root_frequency"] * root_freq_df["sum_gain_root"] / sum_gain
    )
    return root_freq_df[["root_frequency", "weighted_root_frequency"]]


def _calculate_mean_weighted_depth_metric(trees_df: pd.DataFrame):
    return pd.Series(
        trees_df.groupby("split_feature").apply(
            lambda x: np.average(x.depth, weights=x.gain)
        ),
        name="mean_weighted_depth",
    )


def _select_metric(
    importance_full_result: pd.DataFrame, selected_metric: str
):
    variable_importance = importance_full_result[
        ["split_feature", selected_metric]
    ].rename(columns={"split_feature": "Feature", selected_metric: "Value"})
    return variable_importance.sort_values(
        by="Value", ascending=False, ignore_index=True
    )


def _calculate_depth(trees_df: pd.DataFrame, show_progress: bool = False):
    tqdm.pandas(disable=not show_progress)
    trees_df = (
        trees_df.groupby("tree", group_keys=True)
        .progress_apply(_calculate_depth_for_one_tree)
        .reset_index(drop=True)
    )
    return trees_df


def _calculate_depth_for_one_tree(tree):
    non_leaf_nodes = tree.loc[tree["leaf"] == False].index
    for i in non_leaf_nodes:
        if tree.loc[i, "node"] == 0:
            tree.loc[i, "depth"] = 0
        left = tree.loc[i, "left_child"]
        right = tree.loc[i, "right_child"]
        if tree.loc[tree["ID"] == left, "leaf"].values[0] == False:
            tree.loc[tree["ID"] == left, "depth"] = tree.loc[i, "depth"] + 1
        if tree.loc[tree["ID"] == right, "leaf"].values[0] == False:
            tree.loc[tree["ID"] == right, "depth"] = tree.loc[i, "depth"] + 1
    return tree
