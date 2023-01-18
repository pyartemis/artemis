from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot as plt

from artemis._utilities.domain import VisualizationType
from artemis._utilities.exceptions import VisualizationNotSupportedException
from artemis.visualizer._configuration import (
    VisualizationConfiguration,
)


class Visualizer:
    def __init__(self, method: str, vis_config: VisualizationConfiguration):
        self.vis_config = vis_config
        self.method = method

    def accepts(self, vis_type: str) -> bool:
        return vis_type in self.vis_config.accepted_visualizations

    def plot(
        self,
        ovo: pd.DataFrame,
        vis_type: str,
        ova: Optional[pd.DataFrame] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        title: Optional[str] = "default",
        figsize: Tuple[float, float] = (8, 6),
        interactions_ascending_order: bool = False,
        importance_ascending_order: bool = False,
        _full_result: Optional[pd.DataFrame] = None,
        _feature_column_name_1: str = "Feature 1",
        _feature_column_name_2: str = "Feature 2",
        _directed: bool = False,
        **kwargs,
    ):

        if not self.accepts(vis_type):
            raise VisualizationNotSupportedException(self.method, vis_type)

        if vis_type == VisualizationType.SUMMARY:
            self.plot_summary(
                ovo,
                feature_importance,
                ova,
                title=title,
                figsize=figsize,
                interactions_ascending_order=interactions_ascending_order,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualizationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(
                ovo,
                feature_importance,
                title=title,
                figsize=figsize,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                interactions_ascending_order=interactions_ascending_order,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualizationType.BAR_CHART_OVA:
            self.plot_barchart_ova(ova, figsize=figsize, **kwargs)
        elif vis_type == VisualizationType.HEATMAP:
            self.plot_heatmap(
                ovo,
                feature_importance,
                title=title,
                figsize=figsize,
                interactions_ascending_order=interactions_ascending_order,
                importance_ascending_order=importance_ascending_order,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualizationType.BAR_CHART_OVO:
            self.plot_barchart_ovo(
                ovo,
                title=title,
                figsize=figsize,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                **kwargs,
            )
        elif vis_type == VisualizationType.LOLLIPOP:
            self.plot_lollipop(
                _full_result, title=title, figsize=figsize, **kwargs
            )
        elif vis_type == VisualizationType.BAR_CHART_CONDITIONAL:
            self.plot_barchart_conditional(
                ovo,
                feature_importance,
                title=title,
                figsize=figsize,
                **kwargs,
            )

    def plot_barchart_conditional(
            self,
            ovo: pd.DataFrame,
            feature_importance: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            ax=None,
            **kwargs,
    ):
        config = self.vis_config.interaction_bar_chart_conditional
        top_k = kwargs.pop("top_k", config.TOP_K)
        colormap = kwargs.pop("cmap", config.COLOR_MAP)
        color = kwargs.pop("color", config.COLOR)
        title = config.TITLE if title == "default" else title
        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)
        ax.set_axisbelow(True)

        df_to_plot = ovo.copy()
        df_to_plot["Interaction"] = df_to_plot["root_variable"] + ":" + df_to_plot["variable"]
        df_to_plot = df_to_plot.iloc[:top_k]
        df_to_plot = df_to_plot.join(feature_importance.set_index("Feature"), on="variable").rename(
            columns={"Importance": "unconditional_importance"})

        norm = plt.Normalize(df_to_plot["n_occurences"].min(), df_to_plot["n_occurences"].max())
        cmap = plt.get_cmap(colormap)
        new_cmap = colors.LinearSegmentedColormap.from_list("new_cmap", cmap(np.linspace(0.5, 1, 100)))
        sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
        ax = sns.barplot(df_to_plot, x="Interaction", y=self.method, hue="n_occurences",
                         palette=sm.to_rgba(np.flip(df_to_plot["n_occurences"])), dodge=False)
        plt.vlines(x=df_to_plot["Interaction"], ymin=df_to_plot[self.method],
                   ymax=df_to_plot["unconditional_importance"], color=color, alpha=0.5)
        plt.scatter(df_to_plot["Interaction"], y=df_to_plot["unconditional_importance"], color=color, marker="o")
        ax.get_legend().remove()
        plt.colorbar(sm, label="Occurences", ax=ax)
        plt.xticks(rotation=90, size=7)
        plt.grid()

    def plot_heatmap(
            self,
            ovo: pd.DataFrame,
            feature_importance: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            interactions_ascending_order: bool = False,
            importance_ascending_order: bool = False,
            ax=None,
            _f1_name: str = "Feature 1",
            _f2_name: str = "Feature 2",
            _directed: bool = False,
            **kwargs,
    ):
        config = self.vis_config.interaction_matrix
        interaction_color_map = kwargs.pop(
            "interaction_color_map",
            config.INTERACTION_COLOR_MAP if not interactions_ascending_order else config.INTERACTION_COLOR_MAP_REVERSE)
        importance_color_map = kwargs.pop(
            "importance_color_map",
            config.IMPORTANCE_COLOR_MAP if not importance_ascending_order else config.IMPORTANCE_COLOR_MAP_REVERSE
        )
        annot_fmt = kwargs.pop("annot_fmt", config.ANNOT_FMT)
        linewidths = kwargs.pop("linewidths", config.LINEWIDTHS)
        linecolor = kwargs.pop("linecolor", config.LINECOLOR)
        cbar_shrink = kwargs.pop("cbar_shrink", config.CBAR_SHRINK)
        title = config.TITLE if title == "default" else title
        if not _directed:
            ovo_copy = ovo.copy()
            ovo_copy[_f1_name], ovo_copy[_f2_name] = (
                ovo_copy[_f2_name],
                ovo_copy[_f1_name],
            )
            ovo_all_pairs = pd.concat([ovo, ovo_copy])
        else:
            ovo_all_pairs = ovo

        if feature_importance is not None:
            var_imp_diag = self._feature_importance_diag(
                ovo, feature_importance, _f1_name=_f1_name, _f2_name=_f2_name
            )
            ovo_all_pairs = pd.concat([ovo_all_pairs, var_imp_diag])

        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)

        matrix = ovo_all_pairs.pivot_table(
            self.method, _f1_name, _f2_name, aggfunc="first"
        )

        matrix = self.order_heatmap(matrix, asc_order=interactions_ascending_order)

        off_diag_mask = np.eye(matrix.shape[0], dtype=bool)

        sns.heatmap(
            matrix,
            annot=True,
            mask=~off_diag_mask,
            cmap=importance_color_map,
            ax=ax,
            square=True,
            fmt=annot_fmt,
            linewidths=linewidths,
            linecolor=linecolor,
            cbar_kws={"label": "Feature importance", "shrink": cbar_shrink},
        )
        sns.heatmap(
            matrix,
            annot=True,
            mask=off_diag_mask,
            cmap=interaction_color_map,
            ax=ax,
            square=True,
            fmt=annot_fmt,
            linewidths=linewidths,
            linecolor=linecolor,
            cbar_kws={"label": self.method, "shrink": cbar_shrink},
        )

    def plot_interaction_graph(
            self,
            ovo: pd.DataFrame,
            feature_importance: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            ax=None,
            interactions_ascending_order: bool = False,
            _f1_name: str = "Feature 1",
            _f2_name: str = "Feature 2",
            _directed: bool = False,
            **kwargs,
    ):
        config = self.vis_config.interaction_graph
        title = config.TITLE if title == "default" else title
        max_edge_width = kwargs.pop("max_edge_width", config.MAX_EDGE_WIDTH)
        n_highest_with_labels = kwargs.pop(
            "n_highest_with_labels", config.N_HIGHEST_WITH_LABELS
        )
        edge_color = kwargs.pop("edge_color", config.EDGE_COLOR)
        node_color = kwargs.pop("node_color", config.NODE_COLOR)
        node_size = kwargs.pop("node_size", config.NODE_SIZE)
        font_color = kwargs.pop("font_color", config.FONT_COLOR)
        font_weight = kwargs.pop("font_weight", config.FONT_WEIGHT)
        font_size = kwargs.pop("font_size", config.FONT_SIZE)
        threshold_relevant_interaction = kwargs.pop(
            "threshold_relevant_interaction", config.THRESHOLD_RELEVANT_INTERACTION
        )

        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)

        ovo_copy = ovo.copy()

        # filter out non-significant interactions according to the order
        ovo_no_sig_inter = self._filter_insignificant_interactions(interactions_ascending_order,
                                                                   threshold_relevant_interaction, ovo_copy)

        self._interactions_to_edge_widths(interactions_ascending_order, ovo_copy, max_edge_width)

        # return ovo_processed, ovo_no_sig_inter
        G = nx.from_pandas_edgelist(
            ovo_copy,
            source=_f1_name,
            target=_f2_name,
            edge_attr=self.method,
            create_using=nx.DiGraph if _directed else nx.Graph,
        )
        pos = nx.circular_layout(G)

        nx.draw(
            G,
            pos,
            ax=ax,
            width=[G[u][v][self.method] for u, v in G.edges()],
            with_labels=True,
            nodelist=list(feature_importance["Feature"])
            if feature_importance is not None
            else None,
            node_size=[
                node_size * val / np.max(feature_importance["Importance"])
                for val in feature_importance["Importance"]
            ]
            if feature_importance is not None
            else node_size,
            font_size=font_size,
            font_weight=font_weight,
            font_color=font_color,
            node_color=node_color,
            edge_color=edge_color, 
            connectionstyle="arc3,rad=0.3",
        )

        if not interactions_ascending_order:  # no labels for cond, curvy edges do not support it :)
            nx.draw_networkx_edge_labels(
                G,
                pos,
                ax=ax,
                edge_labels=self._edge_labels(
                    ovo_no_sig_inter, _f1_name, _f2_name, n_highest_with_labels
                ),
                font_color=font_color,
                font_weight=font_weight,
            )

    def _interactions_to_edge_widths(self, interactions_ascending_order: bool, ovo_copy: pd.DataFrame,
                                     max_width: float):

        # standardize edge widths by scaling interactions to [0,1] interval
        self._scale_to_unit_interval(ovo_copy)

        # a => (1 - a) for ascending order
        self._reverse_scale_for_ascending(interactions_ascending_order, ovo_copy)

        # map to width
        self._map_to_width(max_width, ovo_copy)

    def plot_barchart_ovo(
            self,
            ovo: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            ax=None,
            _f1_name: str = "Feature 1",
            _f2_name: str = "Feature 2",
            **kwargs,
    ):
        config = self.vis_config.interaction_bar_chart_ovo
        top_k = kwargs.pop("top_k", config.TOP_K)
        color = kwargs.pop("color", config.COLOR)
        title = config.TITLE if title == "default" else title
        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)
        ax.set_axisbelow(True)
        ovo_copy = ovo.copy()
        ovo_copy["Interaction"] = ovo_copy[_f1_name] + ":" + ovo_copy[_f2_name]
        ovo_copy.head(top_k).plot.barh(
            x="Interaction",
            y=self.method,
            xlabel=self.method,
            ylabel="Interaction",
            color=color,
            legend=False,
            grid=True,
            ax=ax,
        )
        plt.gca().invert_yaxis()

    def plot_barchart_ova(
            self,
            ova: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            ax=None,
            **kwargs,
    ):
        config = self.vis_config.interaction_bar_chart_ova
        top_k = kwargs.pop("top_k", config.TOP_K)
        color = kwargs.pop("color", config.COLOR)
        title = config.TITLE if title == "default" else title
        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)
        ax.set_axisbelow(True)
        ova.head(top_k).plot.barh(
            x="Feature",
            y=self.method,
            xlabel=self.method,
            color=color,
            legend=False,
            title=title,
            ax=ax,
            grid=True,
        )
        plt.gca().invert_yaxis()

    def plot_summary(
            self,
            ovo: pd.DataFrame,
            feature_importance: pd.DataFrame,
            ova: Optional[pd.DataFrame] = None,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            interactions_ascending_order: bool = False,
            _f1_name: str = "Feature 1",
            _f2_name: str = "Feature 2",
            _directed: bool = False,
            **kwargs,
    ):
        fig = plt.figure(figsize=tuple(5 * elem for elem in figsize))
        gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        title = f"{self.method} summary" if title == "default" else title

        heatmap_kwargs = kwargs.pop("heatmap", None)
        graph_kwargs = kwargs.pop("graph", None)
        bar_char_ovo_kwargs = kwargs.pop("bar_chart_ovo", None)
        bar_char_ova_kwargs = kwargs.pop("bar_chart_ova", None)

        self.plot_heatmap(
            ovo,
            feature_importance,
            interactions_ascending_order=interactions_ascending_order,
            ax=ax1,
            _f1_name=_f1_name,
            _f2_name=_f2_name,
            _directed=_directed,
            **heatmap_kwargs if heatmap_kwargs is not None else {},
        )
        self.plot_interaction_graph(
            ovo,
            feature_importance,
            ax=ax2,
            _f1_name=_f1_name,
            _f2_name=_f2_name,
            _directed=_directed,
            **graph_kwargs if graph_kwargs is not None else {},
        )
        if ova is not None:
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])
            self.plot_barchart_ovo(
                ovo,
                ax=ax3,
                _f1_name=_f1_name,
                _f2_name=_f2_name,
                **bar_char_ovo_kwargs if bar_char_ovo_kwargs is not None else {},
            )
            self.plot_barchart_ova(
                ova,
                ax=ax4,
                **bar_char_ova_kwargs if bar_char_ova_kwargs is not None else {},
            )
        else:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart_ovo(
                ovo,
                ax=ax3,
                _f1_name=_f1_name,
                _f2_name=_f2_name,
                **bar_char_ovo_kwargs if bar_char_ovo_kwargs is not None else {},
            )

        fig.suptitle(title)

    def plot_lollipop(
            self,
            full_result: pd.DataFrame,
            title: str = "default",
            figsize: Tuple[float, float] = (8, 6),
            ax=None,
            **kwargs,
    ):
        config = self.vis_config.lollipop
        max_trees = kwargs.pop("max_trees", config.MAX_TREES)
        colors = kwargs.pop("colors", config.COLORS)
        markers = kwargs.pop("shapes", config.SHAPES)
        max_depth = kwargs.pop("max_depth", config.MAX_DEPTH)
        label_threshold = kwargs.pop("label_threshold", config.LABEL_THRESHOLD)
        labels = kwargs.pop("labels", config.LABELS)
        scale = kwargs.pop("scale", config.SCALE)
        title = config.TITLE if title == "default" else title
        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)

        full_result_copy = full_result.copy()
        full_result_copy["pair_name"] = (
                full_result_copy["parent_name"] + ":" + full_result_copy["split_feature"]
        )
        roots = full_result_copy.loc[full_result_copy["depth"] == 0].copy()
        roots = roots[roots["tree"] < max_trees * roots["tree"].max()]
        nodes = full_result_copy.loc[
            (full_result_copy["leaf"] == False)
            & (full_result_copy["depth"] > 0)
            & (full_result_copy["depth"] <= max_depth)
            ].copy()
        nodes.loc[full_result_copy["interaction"] == True, "split_feature"] = nodes.loc[
            full_result_copy["interaction"] == True, "pair_name"
        ]
        nodes = nodes[nodes["tree"] < max_trees * nodes["tree"].max()]

        ax.set_axisbelow(True)
        plt.xscale(scale)
        plt.xticks(roots["tree"] + 1, roots["tree"])
        plt.plot(roots["tree"] + 1, roots["gain"], c=colors[0])
        plt.scatter(
            roots["tree"] + 1, roots["gain"], color=colors[0], marker=markers[0]
        )
        for i in range(1, max_depth + 1):
            nodes_to_plot = nodes.loc[nodes["depth"] == i]
            plt.scatter(
                nodes_to_plot["tree"] + 1,
                nodes_to_plot["gain"],
                label=nodes_to_plot["depth"],
                color=colors[i],
                marker=markers[i],
            )
            if labels:
                nodes_to_annotate = nodes_to_plot.loc[
                    nodes_to_plot["interaction"] == True
                    ]
                for j, node in nodes_to_annotate.iterrows():
                    plt.annotate(
                        node["pair_name"],
                        (node["tree"] + 1, node["gain"]),
                        (
                            node["tree"] + 1,
                            node["gain"] + 0.025 * np.max(roots["gain"]),
                        ),
                        rotation=90,
                        size=7,
                    ),

        plt.grid()
        plt.ylim(bottom=0)

        roots_labels = roots.loc[roots["gain"] >= label_threshold * roots["gain"].max()]
        if labels:
            for i, root in roots_labels.iterrows():
                plt.annotate(
                    root["split_feature"],
                    (root["tree"] + 1, root["gain"]),
                    (root["tree"] + 1, root["gain"]),
                )
        plt.xlabel("Tree number")
        plt.ylabel("Gain")

    def _edge_widths(self, G):
        return [
            abs(elem) * self.vis_config.interaction_graph.MAX_EDGE_WIDTH
            for elem in nx.get_edge_attributes(G, self.method).values()
        ]

    def _edge_labels(
            self,
            ovo: pd.DataFrame,
            _f1_name: str,
            _f2_name: str,
            n_highest_with_labels: int,
    ):
        ovo_sorted = ovo.sort_values(by=self.method, ascending=False)
        return {
            (row[_f1_name], row[_f2_name]): round(row[self.method], 2)
            for _, row in ovo_sorted[ovo_sorted[self.method] > 0].head(n_highest_with_labels).iterrows()

        }

    def _feature_importance_diag(
            self,
            ovo,
            feature_importance: pd.DataFrame,
            _f1_name: str = "Feature 1",
            _f2_name: str = "Feature 2",
    ):
        all_features = set(list(ovo[_f1_name]) + list(ovo[_f2_name]))
        var_imp_diag = pd.DataFrame.from_records(
            [
                {
                    _f1_name: f,
                    _f2_name: f,
                    self.method: feature_importance[
                        feature_importance["Feature"] == f
                        ]["Importance"].values[0],
                }
                for f in all_features
            ]
        )
        return var_imp_diag

    def _filter_insignificant_interactions(self,
                                           interactions_ascending_order: bool,
                                           threshold_relevant_interaction: float,
                                           ovo_copy: pd.DataFrame):
        if interactions_ascending_order:
            ovo_copy.loc[ovo_copy[self.method] > threshold_relevant_interaction, self.method] = 0
        else:
            ovo_copy.loc[ovo_copy[self.method] < threshold_relevant_interaction, self.method] = 0

        return ovo_copy.copy()

    def _scale_to_unit_interval(self, ovo_copy: pd.DataFrame):
        vals = ovo_copy[self.method]
        m, M = np.min(vals), np.max(vals)
        ovo_copy.loc[:, self.method] = (vals - m) / (M - m)

    def _reverse_scale_for_ascending(self, interactions_ascending_order: bool, ovo_copy: pd.DataFrame):
        if interactions_ascending_order:
            ovo_copy.loc[ovo_copy[self.method] > 0, self.method] = 1 - ovo_copy.loc[
                ovo_copy[self.method] > 0, self.method]

    def _map_to_width(self, max_width: float, ovo_copy: pd.DataFrame):
        ovo_copy[self.method] = ovo_copy[self.method] * max_width

    @staticmethod
    def order_heatmap(matrix: pd.DataFrame, asc_order: bool) -> pd.DataFrame:
        feature_importance = pd.Series(np.diag(matrix), index=matrix.index)

        lambda_1 = feature_importance.max()
        lambda_2 = matrix.max().max()
        features = list(matrix.index)

        scores = dict()
        for f in features:
            f_copy = features.copy()
            f_copy.remove(f)
            scores[f.lower()] = feature_importance[f] * 1 / lambda_1 + max(matrix[f][f_copy]) * 1 / lambda_2

        ind = matrix.sort_index(key=lambda x: [scores[f_lower] for f_lower in x.str.lower()], ascending=asc_order).index
        m = matrix.reindex(ind)  # re-order rows

        return m[list(ind)]  # re-order columns
