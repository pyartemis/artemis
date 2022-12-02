from typing import Optional

import pandas as pd
from tqdm import tqdm

from ....utilities.exceptions import MethodNotFittedException, MetricNotSupportedException
from ....utilities.domain import InteractionMethod, VisualizationType
from ..._method import FeatureInteractionMethod
from ._handler import GBTreesHandler
from artemis.importance_methods.model_specific import SplitScoreImportance
from artemis.utilities.split_score_metrics import (
    SplitScoreImportanceMetric,
    SplitScoreInteractionMetric,
    _LGBM_UNSUPPORTED_METRICS,
    _ASCENDING_ORDER_METRICS
)
from ._handler import GBTreesHandler
from ..._method import FeatureInteractionMethod
from ....utilities.domain import InteractionMethod


class SplitScoreMethod(FeatureInteractionMethod):
    """Class implementing Split Score Method for extraction of interactions. It applies to gradient boosting tree-based models.
    Currently model from LightGBM and XGBoost packages are supported. 

    Strength of interaction is defined by the metric selected by user (default is sum of gains).

    Attributes:
        metric (str) -- metric used to calculate strength of interaction
        method (str) -- name of interaction method
        visualizer (Visualizer) -- automatically created on the basis of a method and used to create visualizations
        variable_importance (pd.DataFrame) -- variable importance values 
        ovo (pd.DataFrame) -- one versus one variable interaction values 
    
    References:
    - https://modeloriented.github.io/EIX/
    """
    def __init__(self):
        """Constructor for SplitScoreMethod"""
        super().__init__(InteractionMethod.SPLIT_SCORE)
        self.metric = None

    @property
    def interactions_ascending_order(self):
        return self.metric in _ASCENDING_ORDER_METRICS

    def fit(
        self,
        model: GBTreesHandler,
        X: Optional[pd.DataFrame] = None,  # unused as explanations are calculated only for trained model, left for compatibility
        show_progress: bool = False,
        interaction_selected_metric: str = SplitScoreInteractionMetric.MEAN_GAIN,
        importance_selected_metric: str = SplitScoreImportanceMetric.MEAN_GAIN,
        only_def_interactions: bool = True,
    ):
        """Calculates Split Score Interactions and Feature Importance for given model.

        Parameters:
            model (GBTreesHandler) -- model to be explained, can be instance of GBTreesHandler, but if not it will be converted
            X (pd.DataFrame, optional) -- unused as explanations are calculated only for trained model
            show_progress (bool) -- whether to show progress bar 
            interaction_selected_metric (str) -- metric used to calculate strength of interaction, one of ['sum_gain', 'sum_cover', 'mean_gain', 'mean_cover', 'mean_depth']
            importance_selected_metric (str) -- metric used to calculate feature importance, one of ['sum_gain', 'sum_cover', 'mean_gain', 'mean_cover', 'mean_depth', 'mean_weighted_depth', 'root_frequency', 'weighted_root_frequency']
            only_def_interactions (bool) -- whether to return only pair of sequential features that fulfill the definition of interaction (better split score for child feature)
        """
        if not isinstance(model, GBTreesHandler):
            model = GBTreesHandler(model)
        self.metric = interaction_selected_metric
        _check_metrics_with_available_info(model.package, interaction_selected_metric, importance_selected_metric)
        self.full_result = _calculate_full_result(
            model.trees_df, model.package, show_progress
        )
        self.full_ovo = _get_summary(self.full_result, only_def_interactions)
        self.ovo = _get_ovo(self, self.full_ovo, interaction_selected_metric)

        # calculate variable importance
        split_score_importance = SplitScoreImportance()
        self.variable_importance = split_score_importance.importance(
            model=model,
            selected_metric=importance_selected_metric,
            trees_df=model.trees_df,
        )

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """Plots interactions
        
        Parameters:
            vis_type (str) -- type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'summary', 'lolliplot']
            title (str) -- title of plot, default is 'default' which means that title will be automatically generated for selected visualization type
            figsize (tuple) -- size of figure
            show (bool) -- whether to show plot
            **kwargs: additional arguments for plot 
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        self.visualizer.plot(self.ovo,
                             vis_type,
                             variable_importance=self.variable_importance,
                             title=title,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self.interactions_ascending_order,
                             _full_result=self.full_result,
                             **kwargs)


def _calculate_full_result(
    trees_df: pd.DataFrame, model_package: str, show_progress: bool
):
    tqdm.pandas(disable=not show_progress)
    full_result = (
        trees_df.groupby("tree", group_keys=True)
        .progress_apply(_prepare_stats, package=model_package)
        .reset_index(drop=True)
    )
    return full_result[_COLUMNS_TO_CHOSE]


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


def _get_summary(full_result: pd.DataFrame, only_def_interactions: bool = True):
    if only_def_interactions:
        interaction_rows = full_result.loc[full_result["interaction"] == True]
    else:
        interaction_rows = full_result.loc[full_result["depth"] > 0]
    interactions_result = (
        (
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
        )
        .rename(
            columns={
                "parent_name": "parent_variable",
                "split_feature": "child_variable",
            }
        )
        .reset_index(drop=True)
    )
    return interactions_result


def _get_ovo(
    method_class: SplitScoreMethod,
    full_ovo: pd.DataFrame,
    selected_metric: SplitScoreInteractionMetric,
):
    return (
        full_ovo[["parent_variable", "child_variable", selected_metric]]
        .rename(
            columns={
                "parent_variable": "Feature 1",
                "child_variable": "Feature 2",
                selected_metric: method_class.method,
            }
        )
        .sort_values(by=method_class.method, ascending=method_class.interactions_ascending_order, ignore_index=True)
    )

def _check_metrics_with_available_info(package, interaction_selected_metric, importance_selected_metric):
    if package == "lightgbm":
        if interaction_selected_metric in _LGBM_UNSUPPORTED_METRICS:
            raise MetricNotSupportedException(package, interaction_selected_metric)
        if importance_selected_metric in _LGBM_UNSUPPORTED_METRICS:
            raise MetricNotSupportedException(package, importance_selected_metric)

_COLUMNS_TO_CHOSE = [
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
