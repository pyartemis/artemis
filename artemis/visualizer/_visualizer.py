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
    def __init__(
            self,
            method: str,
            vis_config: VisualisationConfiguration
    ):
        self.vis_config = vis_config
        self.method = method

    def accepts(self, vis_type: str) -> bool:
        return vis_type in self.vis_config.accepted_visualisations

    def plot(self,
             ovo: pd.DataFrame,
             vis_type: str,
             ova: Optional[pd.DataFrame] = None,
             variable_importance: Optional[pd.DataFrame] = None, 
             feature_column_name_1: str = "Feature 1", 
             feature_column_name_2: str = "Feature 2",
             directed: bool = False,
             figsize: tuple = (8, 6),  
             show: bool = True, **kwargs):

        if not self.accepts(vis_type):
            raise VisualisationNotSupportedException(self.method, vis_type)

        if vis_type == VisualisationType.SUMMARY:
            self.plot_summary(ovo, variable_importance, ova, figsize = figsize, show = show, f1_name=feature_column_name_1,
                              f2_name=feature_column_name_2, directed=directed, **kwargs)
        elif vis_type == VisualisationType.INTERACTION_GRAPH:
            self.plot_interaction_graph(ovo, variable_importance, figsize = figsize, show = show, f1_name=feature_column_name_1,
                                        f2_name=feature_column_name_2,
                                        directed=directed, **kwargs)
        elif vis_type == VisualisationType.BAR_CHART_OVA:
            self.plot_barchart_ova(ova, figsize = figsize, show = show  **kwargs)
        elif vis_type == VisualisationType.HEATMAP:
            self.plot_heatmap(ovo, variable_importance, figsize = figsize, show = show, f1_name=feature_column_name_1, f2_name=feature_column_name_2,
                              directed=directed, **kwargs)
        elif vis_type == VisualisationType.BAR_CHART_OVO:
            self.plot_barchart_ovo(ovo, figsize = figsize, show = show, f1_name = feature_column_name_1, f2_name = feature_column_name_2, **kwargs)

    def plot_heatmap(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None,
                     f1_name: str = "Feature 1",
                     f2_name: str = "Feature 2", directed: bool = False, **kwargs):
    
        if not directed:
            ovo_copy = ovo.copy()
            ovo_copy[f1_name], ovo_copy[f2_name] = ovo_copy[f2_name], ovo_copy[f1_name]
            ovo_all_pairs = pd.concat([ovo, ovo_copy])
        else:
            ovo_all_pairs = ovo

        if variable_importance is not None:
            var_imp_diag = self._variable_importance_diag(ovo, variable_importance, f1_name=f1_name, f2_name=f2_name)
            ovo_all_pairs = pd.concat([ovo_all_pairs, var_imp_diag])

        fig = None
        if ax is not None:
            ax.set_title(self.vis_config.interaction_matrix.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(self.vis_config.interaction_matrix.TITLE)

        matrix = ovo_all_pairs.pivot_table(self.method, f1_name, f2_name, aggfunc='first')
        off_diag_mask = np.eye(matrix.shape[0], dtype=bool)

        sns.heatmap(matrix, 
                    annot=True, 
                    mask=~off_diag_mask, 
                    cmap=self.vis_config.interaction_matrix.IMPORTANCE_COLOR_MAP,  
                    ax=ax,
                    square=True, 
                    fmt = ".3f",
                    linewidths=0.5, 
                    linecolor='white',
                    cbar_kws={'label': 'Feature importance', "shrink": 0.8})
        sns.heatmap(matrix, 
                    annot=True, 
                    mask=off_diag_mask, 
                    cmap=self.vis_config.interaction_matrix.INTERACTION_COLOR_MAP, 
                    ax=ax,
                    square=True, 
                    fmt = ".3f",
                    linewidths=0.5, 
                    linecolor='white',
                    cbar_kws={'label': self.method, "shrink": 0.8})
        if show: 
            plt.show()
        else:
            return fig 

    def plot_summary(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, ova: Optional[pd.DataFrame] = None,
                     f1_name: str = "Feature 1",
                     f2_name: str = "Feature 2",
                     directed: bool = False,
                     figsize: tuple = (8, 6), show: bool = True,
                     **kwargs):
        nrows = 1 if ova is None else 2
        fig = plt.figure(figsize=(18, nrows * 6))
        gs = gridspec.GridSpec(nrows, 4, hspace=0.4, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])

        self.plot_heatmap(ovo, variable_importance, ax=ax1, f1_name = f1_name, f2_name = f2_name, directed=directed, show=False)
        self.plot_interaction_graph(ovo, variable_importance, ax=ax2, f1_name = f1_name, f2_name = f2_name, directed=directed, show=False)
        if ova is not None:
            ax3 = fig.add_subplot(gs[1, 1:3])
            self.plot_barchart_ova(ova, ax=ax3, show=False)

        fig.suptitle(f"{self.method} summary")
        if show: 
            plt.show()
        else:
            plt.close()
            return fig 


    def plot_interaction_graph(self, ovo: pd.DataFrame, variable_importance: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None,
                               f1_name: str = "Feature 1",
                               f2_name: str = "Feature 2", directed: bool = False,
                               **kwargs):
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
                                    source=f1_name, target=f2_name, edge_attr=self.method,
                                    create_using=nx.DiGraph if directed else nx.Graph)
        pos = nx.spring_layout(G, k=4, weight=self.method, iterations=300)
        nx.draw(
            G,
            pos,
            ax=ax,
            width=self._edge_widths(G),
            with_labels=True,
            nodelist=list(variable_importance["Feature"]) if variable_importance is not None else None,
            node_size=[config.NODE_SIZE * val / max(variable_importance["Value"]) for val in
                       variable_importance["Value"]] if variable_importance is not None else config.NODE_SIZE,
            font_size=config.FONT_SIZE,
            font_weight=config.FONT_WEIGHT,
            font_color=config.FONT_COLOR,
            node_color=config.NODE_COLOR,
            edge_color=self._edge_colors(G),
            connectionstyle="arc3,rad=0.3"
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=self._edge_labels(ovo_copy, f1_name, f2_name),
            font_color=config.FONT_COLOR,
            font_weight=config.FONT_WEIGHT,
        )
        if show: 
            plt.show()
        else:
            return fig 


    def plot_barchart_ovo(self, ovo: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None, f1_name: str = "Feature 1",
                               f2_name: str = "Feature 2",
                               **kwargs):
        config = self.vis_config.interaction_bar_chart_ovo
        fig = None
        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(config.TITLE)
        ax.set_axisbelow(True)
        ovo_copy = ovo.copy()
        ovo_copy["Interaction"] = ovo_copy[f1_name] + ":" + ovo_copy[f2_name]
        ovo_copy.head(config.N_HIGHEST).plot.barh(
            x="Interaction",
            y=self.method,
            xlabel=self.method,
            ylabel="Interaction",
            cmap="crest",
            title=config.TITLE,
            legend=False,
            grid=True,
            ax=ax,
        )
        plt.gca().invert_yaxis()
        if show: 
            plt.show()
        else:
            return fig 


    def plot_barchart_ova(self, ova: pd.DataFrame, figsize: tuple = (8, 6), show: bool = True, ax=None,
                        **kwargs):
        config = self.vis_config.interaction_bar_chart_ova
        fig = None
        if ax is not None:
            ax.set_title(config.TITLE)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plt.title(config.TITLE)
        ax.set_axisbelow(True)
        ova.head(config.N_HIGHEST).plot.barh(
            x="Feature",
            y=self.method,
            ylabel=self.method,
            cmap="crest",
            legend=False,
            title=config.TITLE,
            ax=ax,
            grid=True
        )
        plt.gca().invert_yaxis()
        if show: 
            plt.show()
        else:
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

    def _edge_labels(self, ovo: pd.DataFrame, f1_name: str, f2_name: str):
        return {
            (row[f1_name], row[f2_name]): round(row[self.method], 2)
            for index, row in filter(lambda x: x[1][self.method] > 0,
                                     ovo.head(self.vis_config.interaction_graph.N_HIGHEST_WITH_LABELS).iterrows())
        }

    def _variable_importance_diag(self, ovo, variable_importance: pd.DataFrame, f1_name: str = "Feature 1",
                                  f2_name: str = "Feature 2"):
        all_features = set(list(ovo[f1_name]) + list(ovo[f2_name]))
        var_imp_diag = pd.DataFrame.from_records([{f1_name: f,
                                                   f2_name: f,
                                                   self.method:
                                                       variable_importance[variable_importance["Feature"] == f][
                                                           "Value"].values[0]}
                                                  for f in all_features])
        return var_imp_diag
