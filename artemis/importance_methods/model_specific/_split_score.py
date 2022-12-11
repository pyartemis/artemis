from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from artemis.importance_methods._method import FeatueImportanceMethod
from artemis.interactions_methods.model_specific.gb_trees._handler import GBTreesHandler
from artemis.utilities.domain import ImportanceMethod
from artemis.utilities.split_score_metrics import SplitScoreImportanceMetric


class SplitScoreImportance(FeatueImportanceMethod):
    """
    Split Score Feature Importance.
    It applies to gradient boosting tree-based models.
        It can use data calculated in SplitScore method from `interactions_methods` module and so needs to be calculated together.

    Importance of a feature is defined by the metric selected by user (default is sum of gains).

    Attributes:
    ----------
    method : str 
        Method name.
    feature_importance : pd.DataFrame 
        Feature importance values.
    selected_metric : str
        Metric used for calculating importance.
        
    References:
    ----------
    - https://modeloriented.github.io/EIX/
    """

    def __init__(self):
        """Constructor for SplitScoreImportance"""
        super().__init__(ImportanceMethod.SPLIT_SCORE_IMPORTANCE)
        self.selected_metric = None

    def importance(
            self,
            model,
            features: Optional[List[str]] = None,
            selected_metric: str = SplitScoreImportanceMetric.SUM_GAIN,
            show_progress: bool = False,
            trees_df: Optional[pd.DataFrame] = None,
    ):
        """Calculates Split Score Feature Importance.

        Parameters:
        ----------
        model : object
             Model for which importance will be calculated, should have predict_proba or predict method, or predict_function should be provided. 
        features : List[str], optional
            List of features for which importance will be calculated. If None, all features from X will be used. Default is None.
        selected_metric : str
            Metric used to calculate feature importance, 
            one of ['sum_gain', 'sum_cover', 'mean_gain', 'mean_cover', 'mean_depth', 
            'mean_weighted_depth', 'root_frequency', 'weighted_root_frequency'].
            Default is 'mean_gain'.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        trees_df : pd.DataFrame, optional
            DataFrame containing unified structure of the trained trees, can be precalculated by SplitScore method. Default is None.

        Returns:
        -------
        pd.DataFrame
            Result dataframe containing feature importance with columns: "Feature", "Importance"
        """
        if trees_df is None:
            if not isinstance(model, GBTreesHandler):
                model = GBTreesHandler(model)
            trees_df = model.trees_df

        if trees_df["depth"].isnull().values.any():
            trees_df = _calculate_depth(trees_df, show_progress)
        self.full_result = _calculate_all_feature_importance(
            trees_df, features, selected_metric
        )
        self.feature_importance = _select_metric(self.full_result, selected_metric)
        self.selected_metric = selected_metric

        return self.feature_importance

    @property
    def importance_ascending_order(self):
        return self.selected_metric in [SplitScoreImportanceMetric.MEAN_DEPTH,
                                        SplitScoreImportanceMetric.MEAN_WEIGHTED_DEPTH]


def _calculate_all_feature_importance(
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


def _select_metric(importance_full_result: pd.DataFrame, selected_metric: str):
    feature_importance = importance_full_result[
        ["split_feature", selected_metric]
    ].rename(columns={"split_feature": "Feature", selected_metric: "Importance"})
    return feature_importance.sort_values(
        by="Importance", ascending=False, ignore_index=True
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
