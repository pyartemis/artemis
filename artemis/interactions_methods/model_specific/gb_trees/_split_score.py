from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ....utilities.domain import InteractionMethod
from ..._method import FeatureInteractionMethod
from ._handler import GBTreesHandler
from artemis.importance_methods.model_specific import SplitScoreImportance
from artemis.utilities.split_score_metrics import SplitScoreMetric


class SplitScoreMethod(FeatureInteractionMethod):
    def __init__(self):
        super().__init__(InteractionMethod.SPLIT_SCORE)

    def fit(
        self,
        model: GBTreesHandler,
        X: pd.DataFrame = None,  # unused as explanations are calculated only for trained model, left for compatibility
        show_progress: bool = False,
        selected_metric: SplitScoreMetric = SplitScoreMetric.SUM_GAIN,
        only_def_interactions: bool = True,
    ):
        if not isinstance(model, GBTreesHandler):
            model = GBTreesHandler(model)
        self.full_result = _calculate_full_result(
            model.trees_df, model.package, show_progress
        )
        self.ovo = _get_ovo_summary(self.full_result, only_def_interactions)

        # calculate variable importance
        split_score_importance = SplitScoreImportance()
        self.variable_importance = split_score_importance.importance(
            model = model, 
            selected_metric = selected_metric, 
            trees_df = model.trees_df)


def _calculate_full_result(trees_df: pd.DataFrame, model_package: str, show_progress: bool):
    if show_progress:
        tqdm.pandas()
        full_result = trees_df.groupby("tree", group_keys=True).progress_apply(
            _prepare_stats, model_package
        ).reset_index(drop=True)
    else:
        full_result = trees_df.groupby("tree", group_keys=True).apply(_prepare_stats, model_package).reset_index(drop=True)
    return full_result[
        [
            "tree",
            "ID",
            "depth",
            "split_feature",
            "parent_name",
            "gain",
            "cover",
            "parent_gain",
            "parent_cover",
            "leaf",
            "interaction",
        ]
    ]


def _prepare_stats(tree: pd.DataFrame, package: str):
    non_leaf_nodes = tree.loc[tree["leaf"] == False].index
    for i in non_leaf_nodes:
        if package == "xgboost":
            if tree.loc[i, "node"] == 0:
                tree.loc[i, "depth"] = 0
        left = tree.loc[i, "left_child"]
        right = tree.loc[i, "right_child"]
        if tree.loc[tree["ID"] == left, "leaf"].values[0] == False:
            mask = tree["ID"] == left
            tree.loc[mask, "parent_gain"] = tree.loc[i, "gain"]
            tree.loc[mask, "parent_cover"] = tree.loc[i, "cover"]
            tree.loc[mask, "parent_name"] = tree.loc[i, "split_feature"]
            if package == "xgboost":
                tree.loc[mask, "depth"] = tree.loc[i, "depth"] + 1
            tree.loc[mask, "interaction"] = (
                tree.loc[mask, "parent_gain"] < tree.loc[mask, "gain"]
            ) & (tree.loc[mask, "split_feature"] != tree.loc[mask, "parent_name"])
        if tree.loc[tree["ID"] == right, "leaf"].values[0] == False:
            mask = tree["ID"] == right
            tree.loc[mask, "parent_gain"] = tree.loc[i, "gain"]
            tree.loc[mask, "parent_cover"] = tree.loc[i, "cover"]
            tree.loc[mask, "parent_name"] = tree.loc[i, "split_feature"]
            if package == "xgboost":
                tree.loc[mask, "depth"] = tree.loc[i, "depth"] + 1
            tree.loc[mask, "interaction"] = (
                tree.loc[mask, "parent_gain"] < tree.loc[mask, "gain"]
            ) & (tree.loc[mask, "split_feature"] != tree.loc[mask, "parent_name"])
    return tree


def _get_ovo_summary(full_result: pd.DataFrame, only_def_interactions: bool = True):
    if only_def_interactions:
        interaction_rows = full_result.loc[full_result["interaction"] == True]
    else:
        interaction_rows = full_result.loc[full_result["depth"] > 0]
    interactions_result = (
        interaction_rows.groupby(["parent_name", "split_feature"])
        .agg(
            mean_gain=("gain", "mean"),
            sum_gain=("gain", "sum"),
            mean_cover=("cover", "mean"),
            sum_cover=("cover", "sum"),
            mean_depth=("depth", "mean"),
            frequency=("tree", "count"),
        )
        .reset_index()
        .sort_values("sum_gain", ascending=False)
    ).rename(
        columns={"parent_name": "parent_variable", "split_feature": "child_variable"}
    ).reset_index(drop=True)
    return interactions_result
