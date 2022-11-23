from typing import Optional

import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt

from artemis.utilities.domain import VisualisationType
from artemis.utilities.exceptions import VisualisationNotSupportedException
from artemis.visualisation.configuration import (
    VisualisationConfiguration,
)


class Visualisation:
    def __init__(
            self,
            method: str,
            vis_config: VisualisationConfiguration
    ):
        self.vis_config = vis_config
        self.method = method

    def accepts(self, vis_type: str) -> bool:
        return vis_type in self.vis_config.accepted_visualisations

    def plot(self, ovo: pd.DataFrame, vis_type: str, ova: Optional[pd.DataFrame] = None,
             var_importance: pd.DataFrame = None):

        if not self.accepts(vis_type):
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            self.plot_summary(ovo, var_importance, ova)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(ovo, var_importance)
        elif vis_type == VisualisationType.BAR_CHART:
            self.plot_barchart(ova)
        elif vis_type == VisualisationType.HEATMAP:
            self.plot_heatmap(ovo, var_importance)

    def plot_summary(self, ovo: pd.DataFrame, importance, ova: Optional[pd.DataFrame] = None):
        nrows = 1 if ova is None else 2
        fig = plt.figure(figsize=(18, nrows * 6))
        gs = gridspec.GridSpec(nrows, 4, hspace=0.4, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])

        self.plot_heatmap(ovo, importance, ax1)
        self.plot_interaction_graph(ovo, importance, ax2)

        if ova is not None:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart(ova, ax3)

        fig.suptitle(f"{self.method} summary")

    def plot_heatmap(self, ovo: pd.DataFrame, var_imp, ax=None):
        ovo_copy = ovo.copy()
        ovo_copy["Feature 1"], ovo_copy["Feature 2"] = ovo_copy["Feature 2"], ovo_copy["Feature 1"]
        var_imp_diag = self._variable_importance_diag(ovo, var_imp)
        ovo_all_pairs = pd.concat([ovo, ovo_copy, var_imp_diag])

        if ax is not None:
            ax.set_title(self.vis_config.interaction_matrix.TITLE)
        else:
            plt.title(self.vis_config.interaction_matrix.TITLE)

        sns.heatmap(
            ovo_all_pairs.pivot_table(self.method, "Feature 1", "Feature 2"),
            cmap=self.vis_config.interaction_matrix.COLOR_MAP,
            annot=True,
            ax=ax
        )

    def plot_interaction_graph(self, ovo: pd.DataFrame, var_imp, ax=None):

        ovo_copy = ovo.copy()
        config = self.vis_config.interaction_graph
        ovo_copy.loc[ovo_copy[self.method] < config.MIN_RELEVANT_INTERACTION, self.method] = 0
        G = nx.from_pandas_edgelist(ovo_copy,
                                    source="Feature 1", target="Feature 2", edge_attr=self.method)
        pos = nx.spring_layout(G, k=4, weight=self.method, iterations=300)
        nx.draw(
            G,
            pos,
            ax=ax,
            width=self._edge_widths(G),
            with_labels=True,
            nodelist=var_imp["Feature"],
            node_size=[config.NODE_SIZE * val / max(var_imp["Value"]) for val in var_imp["Value"]],
            font_size=config.FONT_SIZE,
            font_weight=config.FONT_WEIGHT,
            font_color=config.FONT_COLOR,
            node_color=config.NODE_COLOR,
            edge_color=self._edge_colors(G),
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=self._edge_labels(ovo_copy),
            font_color=config.FONT_COLOR,
            font_weight=config.FONT_WEIGHT,
        )

        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            plt.title(config.TITLE)

    def plot_barchart(self, ova: pd.DataFrame, ax=None):
        config = self.vis_config.interaction_bar_chart

        ova.head(config.N_HIGHEST).plot.barh(
            x="Feature",
            y=self.method,
            ylabel=self.method,
            cmap="crest",
            title=config.TITLE,
            ax=ax
        )

    def _edge_widths(self, G):
        return [
            abs(elem) * self.vis_config.interaction_graph.MAX_EDGE_WIDTH
            for elem in nx.get_edge_attributes(G, self.method).values()
        ]

    def _edge_colors(self, G):

        return [
            self.vis_config.interaction_graph.EDGE_COLOR_POS if elem > 0 else self.vis_config.interaction_graph.EDGE_COLOR_NEG
            for elem in
            nx.get_edge_attributes(G, self.method).values()]

    def _edge_labels(self, ovo):
        return {
            (row["Feature 1"], row["Feature 2"]): round(row[self.method], 2)
            for index, row in filter(lambda x: x[1][self.method] > 0,
                                     ovo.head(self.vis_config.interaction_graph.N_HIGHEST_WITH_LABELS).iterrows())
        }

    def _variable_importance_diag(self, ovo, var_imp):
        all_features = set(list(ovo["Feature 1"]) + list(ovo["Feature 2"]))
        var_imp_diag = pd.DataFrame.from_records([{"Feature 1": f,
                                                   "Feature 2": f,
                                                   self.method: var_imp[var_imp["Feature"] == f]["Value"].values[0]}
                                                  for f in all_features])
        return var_imp_diag
