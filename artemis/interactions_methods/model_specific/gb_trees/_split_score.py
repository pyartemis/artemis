from typing import Tuple

import pandas as pd
from tqdm import tqdm

from artemis.importance_methods.model_specific import SplitScoreImportance
from artemis._utilities.split_score_metrics import (
    SplitScoreImportanceMetric,
    SplitScoreInteractionMetric,
    _LGBM_UNSUPPORTED_METRICS,
    _ASCENDING_ORDER_METRICS
)
from artemis._utilities._handler import GBTreesHandler
from ..._method import FeatureInteractionMethod
from ...._utilities.domain import InteractionMethod
from ...._utilities.domain import VisualizationType
from ...._utilities.exceptions import MethodNotFittedException, MetricNotSupportedException


class SplitScoreMethod(FeatureInteractionMethod):
    """
    Split Score Method for Feature Interaction Extraction.
    It applies to gradient boosting tree-based models.
    Currently model from LightGBM and XGBoost packages are supported. 

    Strength of interaction is defined by the metric selected by user (default is sum of gains).

    Attributes
    ----------
    method : str 
        Method name, used also for naming column with results in `ovo` pd.DataFrame.
    visualizer : Visualizer
        Object providing visualization. Automatically created on the basis of a method and used to create visualizations.
    ovo : pd.DataFrame 
        One versus one (pair) feature interaction values. 
    feature_importance : pd.DataFrame 
        Feature importance values.
    model : object
        Explained model.
    metric : str 
        Metric used to calculate strength of interactions.
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.

    References
    ----------
    - https://modeloriented.github.io/EIX/
    """
    def __init__(self):
        """Constructor for SplitScoreMethod"""
        super().__init__(InteractionMethod.SPLIT_SCORE)
        self.metric = None

    @property
    def _interactions_ascending_order(self):
        return self.metric in _ASCENDING_ORDER_METRICS

    def fit(
        self,
        model: GBTreesHandler,
        show_progress: bool = False,
        interaction_selected_metric: str = SplitScoreInteractionMetric.MEAN_GAIN,
        importance_selected_metric: str = SplitScoreImportanceMetric.MEAN_GAIN,
        only_def_interactions: bool = True,
    ):
        """Calculates Split Score Feature Interactions Strength and Split Score Feature Importance for given model.

        Parameters
        ----------
        model : GBTreesHandler
            Model to be explained. Should be fitted and of type GBTreesHandler (otherwise it will be converted). 
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        interaction_selected_metric : str 
            Metric used to calculate strength of interaction, 
            one of ['sum_gain', 'sum_cover', 'mean_gain', 'mean_cover', 'mean_depth']. Default is 'mean_gain'.
        importance_selected_metric : str 
            Metric used to calculate feature importance, 
            one of ['sum_gain', 'sum_cover', 'mean_gain', 'mean_cover', 'mean_depth', 
            'mean_weighted_depth', 'root_frequency', 'weighted_root_frequency'].
            Default is 'mean_gain'.
        only_def_interactions : bool 
            Whether to return only pair of sequential features that fulfill the definition of interaction 
            (with better split score for child feature).
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

        # calculate feature importance
        self._feature_importance_obj = SplitScoreImportance()
        self.feature_importance = self._feature_importance_obj.importance(
            model=model,
            selected_metric=importance_selected_metric,
            trees_df=model.trees_df,
        )

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: Tuple[float, float] = (8, 6), **kwargs):
        """
        Plot results of explanations.

        There are five types of plots available:
        - heatmap - heatmap of feature interactions values with feature importance values on the diagonal (default)
        - bar_chart - bar chart of top feature interactions values
        - graph - graph of feature interactions values
        - summary - combination of heatmap, bar chart and graph plots
        - lolliplot - lolliplot for first k decision trees with split scores values.
        
        Parameters
        ----------
        vis_type : str 
            Type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'lolliplot', 'summary']. Default is 'heatmap'.
        title : str 
            Title of plot, default is 'default' which means that title will be automatically generated for selected visualization type.
        figsize : (float, float) 
            Size of plot. Default is (8, 6).
        **kwargs : Other Parameters
            Additional parameters for plot. Passed to suitable matplotlib or seaborn functions. 
            For 'summary' visualization parameters for respective plots should be in dict with keys corresponding to visualization name. 
            See key parameters below. 
        
        Other Parameters
        ------------------------
        interaction_color_map : matplotlib colormap name or object, or list of colors
            Used for 'heatmap' visualization. The mapping from interaction values to color space. Default is 'Purples' or 'Purpler_r',
            depending on whether a greater value means a greater interaction strength or vice versa.
        importance_color_map :  matplotlib colormap name or object, or list of colors
            Used for 'heatmap' visualization. The mapping from importance values to color space. Default is 'Greens' or 'Greens_r',
            depending on whether a greater value means a greater interaction strength or vice versa.
        annot_fmt : str
            Used for 'heatmap' visualization. String formatting code to use when adding annotations with values. Default is '.3f'.
        linewidths : float
            Used for 'heatmap' visualization. Width of the lines that will divide each cell in matrix. Default is 0.5.
        linecolor : str
            Used for 'heatmap' visualization. Color of the lines that will divide each cell in matrix. Default is 'white'.
        cbar_shrink : float
            Used for 'heatmap' visualization. Fraction by which to multiply the size of the colorbar. Default is 1. 
    
        top_k : int 
            Used for 'bar_chart' visualization. Maximum number of pairs that will be presented in plot. Default is 10.
        color : str 
            Used for 'bar_chart' visualization. Color of bars. Default is 'mediumpurple'.

        n_highest_with_labels : int
            Used for 'graph' visualization. Top most important interactions to show as labels on edges.  Default is 5.
        edge_color: str
            Used for 'graph' visualization. Color of the edges. Default is 'rebeccapurple.
        node_color: str
            Used for 'graph' visualization. Color of nodes. Default is 'green'.
        node_size: int
            Used for 'graph' visualization. Size of the nodes (networkX scale).  Default is '1800'.
        font_color: str
            Used for 'graph' visualization. Font color. Default is '#3B1F2B'.
        font_weight: str
            Used for 'graph' visualization. Font weight. Default is 'bold'.
        font_size: int
            Used for 'graph' visualization. Font size (networkX scale). Default is 10.
        threshold_relevant_interaction : float
            Used for 'graph' visualization. Minimum (or maximum, depends on method) value of interaction to display
            corresponding edge on visualization. Default depends on the interaction method.

        max_trees : float
            Used for 'lolliplot' visualization. Fraction of trees that will be presented in plot. Default is 0.2. 
        colors : List[str]
            Used for 'lolliplot' visualization. List of colors for nodes with successive depths. 
            Default is ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]. 
        shapes : List[str]
            Used for 'lolliplot' visualization. List of shapes for nodes with successive depths. 
            Default is ["o", ",", "v", "^", "<", ">"]. 
        max_depth : int
            Used for 'lolliplot' visualization. Threshold for depth of nodes that will be presented in plot. Default is 1. 
        label_threshold : float
            Used for 'lolliplot' visualization. Threshold for fraction of score of nodes that will be labeled in plot. Default is 0.1. 
        labels : bool
            Used for 'lolliplot' visualization. Whether to add labels to plot. Default is True. 
        scale : str
            Used for 'lolliplot' visualization. Scale for x axis (trees). Default is 'linear'. 
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        self.visualizer.plot(self.ovo,
                             vis_type,
                             feature_importance=self.feature_importance,
                             title=title,
                             figsize=figsize,
                             interactions_ascending_order=self._interactions_ascending_order,
                             importance_ascending_order=self._feature_importance_obj.importance_ascending_order,
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
                "parent_name": "parent_feature",
                "split_feature": "child_feature",
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
        full_ovo[["parent_feature", "child_feature", selected_metric]]
        .rename(
            columns={
                "parent_feature": "Feature 1",
                "child_feature": "Feature 2",
                selected_metric: method_class.method,
            }
        )
        .sort_values(by=method_class.method, ascending=method_class._interactions_ascending_order, ignore_index=True)
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
