from typing import Optional

import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

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

    def plot(self, ovo: pd.DataFrame, vis_type: str, ova: Optional[pd.DataFrame] = None):

        if not self.accepts(vis_type):
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            self.plot_summary(ovo, ova)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(ovo)
        elif vis_type == VisualisationType.BAR_CHART:
            self.plot_barchart(ova)
        elif vis_type == VisualisationType.HEATMAP:
            self.plot_heatmap(ovo)

    def plot_summary(self, ovo: pd.DataFrame, ova: Optional[pd.DataFrame] = None):
        nrows = 1 if ova is None else 2
        fig = plt.figure(figsize=(18, nrows * 6))
        gs = gridspec.GridSpec(nrows, 4, hspace=0.4, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])

        self.plot_heatmap(ovo, ax1)
        self.plot_interaction_graph(ovo, ax2)

        if ova is not None:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart(ova, ax3)

        fig.suptitle(f"{self.method} summary")

    def plot_heatmap(self, ovo: pd.DataFrame, ax=None):
        ovo_copy = ovo.copy()
        ovo_copy["Feature 1"], ovo_copy["Feature 2"] = ovo_copy["Feature 2"], ovo_copy["Feature 1"]
        ovo_all_pairs = pd.concat([ovo, ovo_copy])

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

    def plot_interaction_graph(self, ovo: pd.DataFrame, ax=None):
        config = self.vis_config.interaction_graph
        ovo_relevant_interactions = ovo[abs(ovo[self.method]) > config.MIN_RELEVANT_INTERACTION]
        G = nx.from_pandas_edgelist(ovo_relevant_interactions,
                                    source="Feature 1", target="Feature 2", edge_attr=self.method)
        pos = nx.spring_layout(G, weight=self.method, iterations=300)
        nx.draw(
            G,
            pos,
            ax=ax,
            width=self._edge_widths(G),
            with_labels=True,
            node_size=config.NODE_SIZE,
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
            edge_labels=self._edge_labels(ovo_relevant_interactions),
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
            for index, row in ovo.head(self.vis_config.interaction_graph.N_HIGHEST_WITH_LABELS).iterrows()
        }
