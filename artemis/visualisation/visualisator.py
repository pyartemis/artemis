from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt

from artemis.utilities.domain import VisualisationType
from artemis.utilities.exceptions import VisualisationNotSupportedException
from artemis.visualisation.configuration import (
    VisualisationConfiguration,
)


class Visualizator:
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
             variable_importance: Optional[pd.DataFrame] = None, figsize: tuple = (8, 6),  
             show: bool = True, **kwargs):

        if not self.accepts(vis_type):
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            self.plot_summary(ovo, variable_importance, ova, figsize = figsize, show = show)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(ovo, variable_importance, figsize = figsize, show = show)
        elif vis_type == VisualisationType.BAR_CHART_OVA:
            self.plot_barchart_ova(ova, figsize = figsize, show = show)
        elif vis_type == VisualisationType.HEATMAP:
            self.plot_heatmap(ovo, variable_importance, figsize = figsize, show = show)
        elif vis_type == VisualisationType.BAR_CHART_OVO:
            self.plot_barchart_ovo(ovo, figsize = figsize, show = show)

    def plot_heatmap(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None):
        ovo_copy = ovo.copy()
        ovo_copy["Feature 1"], ovo_copy["Feature 2"] = ovo_copy["Feature 2"], ovo_copy["Feature 1"]
        var_imp_diag = self._variable_importance_diag(ovo, variable_importance)
        ovo_all_pairs = pd.concat([ovo, ovo_copy, var_imp_diag])
        fig = None
        if ax is not None:
            ax.set_title(self.vis_config.interaction_matrix.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(self.vis_config.interaction_matrix.TITLE)

        matrix = ovo_all_pairs.pivot_table(self.method, "Feature 1", "Feature 2", aggfunc='first')
        off_diag_mask = np.eye(matrix.shape[0], dtype=bool)

        sns.heatmap(matrix, 
                    annot=True, 
                    mask=~off_diag_mask, 
                    cmap=self.vis_config.interaction_matrix.IMPORTANCE_COLOR_MAP,  
                    ax=ax)
        sns.heatmap(matrix, 
                    annot=True, 
                    mask=off_diag_mask, 
                    cmap=self.vis_config.interaction_matrix.INTERACTION_COLOR_MAP, 
                    ax=ax)
        if show: 
            plt.show()
        else:
            plt.close()
            return fig 

    def plot_summary(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, ova: Optional[pd.DataFrame] = None, figsize: tuple = (8, 6), show: bool = True):
        nrows = 1 if ova is None else 2
        fig = plt.figure(figsize=(18, nrows * 6))
        gs = gridspec.GridSpec(nrows, 4, hspace=0.4, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])

        self.plot_heatmap(ovo, variable_importance, ax=ax1, show=False)
        self.plot_interaction_graph(ovo, variable_importance, ax=ax2, show=False)

        if ova is not None:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart_ova(ova, ax=ax3, show=False)

        fig.suptitle(f"{self.method} summary")
        if show: 
            plt.show()
        else:
            plt.close()
            return fig 


    def plot_interaction_graph(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None):
        config = self.vis_config.interaction_graph
        fig = None
        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(config.TITLE)
        ovo_copy = ovo.copy()
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
            nodelist=variable_importance["Feature"],
            node_size=[config.NODE_SIZE * val / max(variable_importance["Value"]) for val in
                       variable_importance["Value"]],
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
        if show: 
            plt.show()
        else:
            plt.close()
            return fig 



    def plot_barchart_ovo(self, ovo: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None):
        config = self.vis_config.interaction_bar_chart_ovo
        fig = None
        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(config.TITLE)
        ovo_copy = ovo.copy()
        ovo_copy["Interaction"] = ovo_copy["Feature 1"] + ":" + ovo_copy["Feature 2"]

        ovo_copy.head(config.N_HIGHEST).plot.barh(
            x="Interaction",
            y=self.method,
            xlabel=self.method,
            ylabel="Interaction",
            cmap="crest",
            title=config.TITLE,
            ax=ax
        )
        plt.gca().invert_yaxis()
        if show: 
            plt.show()
        else:
            plt.close()
            return fig 


    def plot_barchart_ova(self, ova: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None):
        config = self.vis_config.interaction_bar_chart_ova
        fig = None
        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(config.TITLE)
        ova.head(config.N_HIGHEST).plot.barh(
            x="Feature",
            y=self.method,
            ylabel=self.method,
            cmap="crest",
            title=config.TITLE,
            ax=ax
        )
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

    def _variable_importance_diag(self, ovo, variable_importance: pd.DataFrame):
        all_features = set(list(ovo["Feature 1"]) + list(ovo["Feature 2"]))
        var_imp_diag = pd.DataFrame.from_records([{"Feature 1": f,
                                                   "Feature 2": f,
                                                   self.method:
                                                       variable_importance[variable_importance["Feature"] == f][
                                                           "Value"].values[0]}
                                                  for f in all_features])
        return var_imp_diag
