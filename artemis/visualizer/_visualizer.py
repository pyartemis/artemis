from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

from artemis.utilities.domain import VisualisationType
from artemis.utilities.exceptions import VisualisationNotSupportedException
from artemis.visualizer._configuration import (
    VisualisationConfiguration,
)


class Visualizer:
    def __init__(self, method: str, vis_config: VisualisationConfiguration):
        self.vis_config = vis_config
        self.method = method

    def accepts(self, vis_type: str) -> bool:
        return vis_type in self.vis_config.accepted_visualisations

    def plot(
        self,
        ovo: pd.DataFrame,
        vis_type: str,
        ova: Optional[pd.DataFrame] = None,
        variable_importance: Optional[pd.DataFrame] = None,
        title: Optional[str] = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
        _feature_column_name_1: str = "Feature 1",
        _feature_column_name_2: str = "Feature 2",
        _directed: bool = False,
        **kwargs,
    ):

        if not self.accepts(vis_type):
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            self.plot_summary(
                ovo,
                variable_importance,
                ova,
                title=title,
                figsize=figsize,
                show=show,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(
                ovo,
                variable_importance,
                title=title,
                figsize=figsize,
                show=show,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualisationType.BAR_CHART_OVA:
            self.plot_barchart_ova(ova, figsize=figsize, show=show, **kwargs)
        elif vis_type == VisualisationType.HEATMAP:
            self.plot_heatmap(
                ovo,
                variable_importance,
                title=title,
                figsize=figsize,
                show=show,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                _directed=_directed,
                **kwargs,
            )
        elif vis_type == VisualisationType.BAR_CHART_OVO:
            self.plot_barchart_ovo(
                ovo,
                title=title,
                figsize=figsize,
                show=show,
                _f1_name=_feature_column_name_1,
                _f2_name=_feature_column_name_2,
                **kwargs,
            )

    def plot_heatmap(
        self,
        ovo: pd.DataFrame,
        variable_importance: pd.DataFrame,
        title: str = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
        ax=None,
        _f1_name: str = "Feature 1",
        _f2_name: str = "Feature 2",
        _directed: bool = False,
        **kwargs,
    ):
        config = self.vis_config.interaction_matrix
        interaction_color_map = kwargs.pop(
            "interaction_color_map", config.INTERACTION_COLOR_MAP
        )
        importance_color_map = kwargs.pop(
            "importance_color_map", config.IMPORTANCE_COLOR_MAP
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

        if variable_importance is not None:
            var_imp_diag = self._variable_importance_diag(
                ovo, variable_importance, _f1_name=_f1_name, _f2_name=_f2_name
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
        if show:
            plt.show()
        else:
            return fig

    def plot_interaction_graph(
        self,
        ovo: pd.DataFrame,
        variable_importance: pd.DataFrame,
        title: str = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
        ax=None,
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
        edge_color_pos = kwargs.pop("edge_color_pos", config.EDGE_COLOR_POS)
        edge_color_neg = kwargs.pop("edge_color_neg", config.EDGE_COLOR_NEG)
        node_color = kwargs.pop("node_color", config.NODE_COLOR)
        node_size = kwargs.pop("node_size", config.NODE_SIZE)
        font_color = kwargs.pop("font_color", config.FONT_COLOR)
        font_weight = kwargs.pop("font_weight", config.FONT_WEIGHT)
        font_size = kwargs.pop("font_size", config.FONT_SIZE)
        min_relevant_interaction = kwargs.pop(
            "min_relevant_interaction", config.MIN_RELEVANT_INTERACTION
        )

        fig = None
        if ax is not None:
            ax.set_title(title)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(title)

        ovo_copy = ovo.copy()
        ovo_copy.loc[ovo_copy[self.method] < min_relevant_interaction, self.method] = 0
        G = nx.from_pandas_edgelist(
            ovo_copy,
            source=_f1_name,
            target=_f2_name,
            edge_attr=self.method,
            create_using=nx.DiGraph if _directed else nx.Graph,
        )
        pos = nx.spring_layout(G, k=4, weight=self.method, iterations=300)
        nx.draw(
            G,
            pos,
            ax=ax,
            width=[
                max_edge_width * val / max(ovo_copy[self.method])
                for val in ovo_copy[self.method]
            ],
            with_labels=True,
            nodelist=list(variable_importance["Feature"])
            if variable_importance is not None
            else None,
            node_size=[
                node_size * val / max(variable_importance["Value"])
                for val in variable_importance["Value"]
            ]
            if variable_importance is not None
            else node_size,
            font_size=font_size,
            font_weight=font_weight,
            font_color=font_color,
            node_color=node_color,
            edge_color=edge_color,#self._edge_colors(G, edge_color_pos, edge_color_neg),
            connectionstyle="arc3,rad=0.3",
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=self._edge_labels(ovo_copy, _f1_name, _f2_name, n_highest_with_labels),
            font_color=font_color,
            font_weight=font_weight,
        )
        if show:
            plt.show()
        else:
            return fig

    def plot_barchart_ovo(
        self,
        ovo: pd.DataFrame,
        title: str = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
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
        if show:
            plt.show()
        else:
            return fig

    def plot_barchart_ova(
        self,
        ova: pd.DataFrame,
        title: str = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
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
            ylabel=self.method,
            color=color,
            legend=False,
            title=title,
            ax=ax,
            grid=True,
        )
        plt.gca().invert_yaxis()
        if show:
            plt.show()
        else:
            return fig

    def plot_summary(
        self,
        ovo: pd.DataFrame,
        variable_importance: pd.DataFrame,
        ova: Optional[pd.DataFrame] = None,
        title: str = "default",
        figsize: tuple = (8, 6),
        show: bool = True,
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
            variable_importance,
            ax=ax1,
            _f1_name=_f1_name,
            _f2_name=_f2_name,
            _directed=_directed,
            show=False,
            **heatmap_kwargs if heatmap_kwargs is not None else {},
        )
        self.plot_interaction_graph(
            ovo,
            variable_importance,
            ax=ax2,
            _f1_name=_f1_name,
            _f2_name=_f2_name,
            _directed=_directed,
            show=False,
            **graph_kwargs if graph_kwargs is not None else {},
        )
        if ova is not None:
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])
            self.plot_barchart_ovo(
                ovo,
                ax=ax3,
                show=False,
                _f1_name=_f1_name,
                _f2_name=_f2_name,
                **bar_char_ovo_kwargs if bar_char_ovo_kwargs is not None else {},
            )
            self.plot_barchart_ova(
                ova,
                ax=ax4,
                show=False,
                **bar_char_ova_kwargs if bar_char_ova_kwargs is not None else {},
            )
        else:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart_ovo(
                ovo,
                ax=ax3,
                show=False,
                _f1_name=_f1_name,
                _f2_name=_f2_name,
                **bar_char_ovo_kwargs if bar_char_ovo_kwargs is not None else {},
            )

        fig.suptitle(title)
        if show:
            plt.show()
        else:
            plt.close()
            return fig

    def _edge_widths(self, G):
        return [
            abs(elem) * self.vis_config.interaction_graph.MAX_EDGE_WIDTH
            for elem in nx.get_edge_attributes(G, self.method).values()
        ]

    def _edge_colors(self, G, edge_color_pos, edge_color_neg):
        return [
            edge_color_pos
            if elem > 0
            else edge_color_neg
            for elem in nx.get_edge_attributes(G, self.method).values()
        ]

    def _edge_labels(self, ovo: pd.DataFrame, _f1_name: str, _f2_name: str, n_highest_with_labels: int):
        return {
            (row[_f1_name], row[_f2_name]): round(row[self.method], 2)
            for index, row in filter(
                lambda x: x[1][self.method] > 0,
                ovo.head(
                    n_highest_with_labels
                ).iterrows(),
            )
        }

    def _variable_importance_diag(
        self,
        ovo,
        variable_importance: pd.DataFrame,
        _f1_name: str = "Feature 1",
        _f2_name: str = "Feature 2",
    ):
        all_features = set(list(ovo[_f1_name]) + list(ovo[_f2_name]))
        var_imp_diag = pd.DataFrame.from_records(
            [
                {
                    _f1_name: f,
                    _f2_name: f,
                    self.method: variable_importance[
                        variable_importance["Feature"] == f
                    ]["Value"].values[0],
                }
                for f in all_features
            ]
        )
        return var_imp_diag
