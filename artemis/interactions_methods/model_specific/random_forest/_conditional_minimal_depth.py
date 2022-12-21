from collections import defaultdict
from itertools import combinations
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tqdm import tqdm

from artemis.importance_methods.model_specific import MinimalDepthImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis._utilities.domain import InteractionMethod, VisualizationType
from artemis._utilities.exceptions import MethodNotFittedException


class ConditionalMinimalDepthMethod(FeatureInteractionMethod):
    """
    Conditional Smallest Depth Method for Feature Interaction Extraction.
    It applies to tree-based models like Random Forests.
    Currently scikit-learn forest models are supported, i.e., RandomForestClassifier, RandomForestRegressor, 
    ExtraTreesRegressor, ExtraTreesClassifier. 

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
    features_included: List[str]
        List of features for which interactions are calculated.
    pairs : List[List[str]]
        List of pairs of features for which interactions are calculated.

    References
    ----------
    - https://modeloriented.github.io/randomForestExplainer/
    - https://doi.org/10.1198/jasa.2009.tm08622
    """
    def __init__(self):
        """Constructor for ConditionalMinimalDepthMethod"""
        super().__init__(InteractionMethod.CONDITIONAL_MINIMAL_DEPTH)

    @property
    def _interactions_ascending_order(self):
        return True

    @property
    def _compare_ovo(self):
        if self.ovo is None:
            raise MethodNotFittedException(self.method)
        compare_ovo = self.ovo.copy().rename(columns={"root_variable": "Feature 1", "variable": "Feature 2"})
        compare_ovo['id'] = compare_ovo[["Feature 1", "Feature 2"]].apply(lambda x: "".join(sorted(x)), axis=1)
        return (compare_ovo.groupby("id")
                           .agg({"Feature 1": "first", "Feature 2": "first", self.method: "mean"})
                           .sort_values("Conditional Smallest Depth Measure",
                                        ascending=self._interactions_ascending_order,
                                        ignore_index=True))


    def fit(
            self,
            model: Union[RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier],
            show_progress: bool = False,
    ):
        """Calculates Conditional Smallest Depth Feature Interactions Strenght and Minimal Depth Feature Importance for given model.

        Parameters
        ----------
        model : RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, or ExtraTreesClassifier
            Model to be explained. Should be fitted and of type RandomForestClassifier, RandomForestRegressor, 
            ExtraTreesRegressor, or ExtraTreesClassifier.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        """
        self.features_included = model.feature_names_in_.tolist()
        self.pairs = list(combinations(self.features_included, 2))
        column_dict = _make_column_dict(model.feature_names_in_)
        self.raw_result_df, trees = _calculate_conditional_minimal_depths(model.estimators_, len(model.feature_names_in_), show_progress)
        self.ovo = _summarise_results(self.raw_result_df, column_dict, self.method, self._interactions_ascending_order)
        self._feature_importance_obj = MinimalDepthImportance()
        self.feature_importance = self._feature_importance_obj.importance(model,trees)

    def plot(self, vis_type: str = VisualizationType.HEATMAP, title: str = "default", figsize: tuple = (8, 6), show: bool = True, **kwargs):
        """
        Plot results of explanations.

        There are five types of plots available:
        - heatmap - heatmap of feature interactions values with feature importance values on the diagonal (default)
        - bar_chart - bar chart of top feature interactions values
        - graph - graph of feature interactions values
        - summary - combination of heatmap, bar chart and graph plots
        - bar_chart_conditional - bar chart of top feature interactions with additional information about feature importance
        
        Parameters
        ----------
        vis_type : str 
            Type of visualization, one of ['heatmap', 'bar_chart', 'graph', 'bar_chart_conditional', 'summary']. Default is 'heatmap'.
        title : str 
            Title of plot, default is 'default' which means that title will be automatically generated for selected visualization type.
        figsize : (float, float) 
            Size of plot. Default is (8, 6).
        show : bool 
            Whether to show plot. Default is True.
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
        
        top_k : int 
            Used for 'bar_chart_conditional' visualization. Maximum number of pairs that will be presented in plot. Default is 15. 
        cmap : matplotlib colormap name or object.
            Used for 'bar_chart_conditional' visualization. The mapping from number of pair occurences to color space. Default is 'Purples'. 
        color : str
            Used for 'bar_chart_conditional' visualization. Color of lollipops for parent features. Default is 'black'. 
        """
        if self.ovo is None:
            raise MethodNotFittedException(self.method)

        self.visualizer.plot(self.ovo,
                             vis_type,
                             _feature_column_name_1="root_variable",
                             _feature_column_name_2="variable",
                             _directed=True,
                             feature_importance=self.feature_importance,
                             title = title,
                             figsize=figsize,
                             show=show,
                             interactions_ascending_order=self._interactions_ascending_order,
                             importance_ascending_order=self._feature_importance_obj.importance_ascending_order,
                             **kwargs)


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

    Returns
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
    Returns
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
            occurence_flag = 0
            if f_1 in split_var_to_nodes and f_2 in split_var_to_nodes:
                split_f1, split_f2 = split_var_to_nodes[f_1], split_var_to_nodes[f_2]
                highest_maximal_split_f1 = split_f1[depths[split_f1] == depths[split_f1[0]]]
                for current_root_id in highest_maximal_split_f1:
                    current_root_depth = depths[current_root_id]
                    upper_bound = _calculate_maximal_id_in_subtree(current_root_depth, current_root_id, depths, n_nodes)
                    id_f2_in_subtree = _f_2_split_nodes_in_f_1_subtree(current_root_id, split_f2, upper_bound)
                    if len(id_f2_in_subtree) > 0:
                        min_val = min(min_val, np.min(depths[split_f2[id_f2_in_subtree]]) - current_root_depth - 1)
                        occurence_flag = 1
                    else:
                        min_val = min(min_val, depths[upper_bound] - current_root_depth)
            conditional_depths.append({"split_variable": f_1, "ancestor_variable": f_2, "value": min_val, "occur": occurence_flag})

    res = pd.DataFrame.from_records(conditional_depths).replace({float("+inf"): None})

    return res


def _make_column_dict(columns: np.ndarray) -> dict:
    return dict(zip(range(len(columns)), list(columns)))


def _summarise_results(raw_result_df: pd.DataFrame, column_dict: dict, method_name: str, ascending_order: bool) -> pd.DataFrame:
    """Average result over trees, rename columns, fill missing data."""
    final_result = (raw_result_df.groupby(["split_variable", "ancestor_variable"])
                    .agg({"value": "mean", "occur": "sum"})
                    .reset_index()
                    .rename({"occur": "n_occurences", "value": method_name,
                                "split_variable": "root_variable", "ancestor_variable": "variable"}, axis=1)
                    .sort_values(by=["n_occurences", method_name], ascending=[False, ascending_order])
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
    upper_bound = n_nodes-1

    if len(upper_bound_set) > 0:
        upper_bound = np.min(upper_bound_set)-1

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
    return np.where((split_f2 > current_root_id) & (split_f2 <= upper_bound))[0]
