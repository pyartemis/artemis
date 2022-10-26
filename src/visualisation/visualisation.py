from typing import Optional

import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

from src.visualisation.configuration import (
    InteractionGraphConfiguration,
    InteractionMatrixConfiguration,
    InteractionVersusAllConfiguration,
)


class Visualisation:
    def __init__(
            self,
            method: str,
            matrix_config: InteractionMatrixConfiguration = InteractionMatrixConfiguration.default(),
            graph_config: InteractionGraphConfiguration = InteractionGraphConfiguration.default(),
            versus_all_config: InteractionVersusAllConfiguration = InteractionVersusAllConfiguration.default(),
    ):
        self.method = method
        self.matrix_config = matrix_config
        self.graph_config = graph_config
        self.versus_all_config = versus_all_config

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
            ax.set_title(self.matrix_config.TITLE)
        else:
            plt.title(self.matrix_config.TITLE)

        sns.heatmap(
            ovo_all_pairs.pivot_table(self.method, "Feature 1", "Feature 2"),
            cmap=self.matrix_config.COLOR_MAP,
            annot=True,
            ax=ax
        )

    def plot_interaction_graph(self, ovo: pd.DataFrame, ax=None):
        config = self.graph_config
        ovo_relevant_interactions = ovo[ovo[self.method] > self.graph_config.MIN_RELEVANT_INTERACTION]
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
            edge_color=config.EDGE_COLOR,
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
            ax.set_title(self.graph_config.TITLE)
        else:
            plt.title(self.graph_config.TITLE)

    def plot_barchart(self, ova: pd.DataFrame, ax=None):
        config = self.versus_all_config

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
            elem * self.graph_config.MAX_EDGE_WIDTH
            for elem in nx.get_edge_attributes(G, self.method).values()
        ]

    def _edge_labels(self, ovo):
        return {
            (row["Feature 1"], row["Feature 2"]): round(row[self.method], 2)
            for index, row in ovo.head(self.graph_config.N_HIGHEST_WITH_LABELS).iterrows()
        }
