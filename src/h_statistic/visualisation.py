import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, gridspec

from src.domain.domain import ONE_VS_ALL
from src.h_statistic.calculation import calculate_h_stat
from src.h_statistic.configuration import InteractionGraphConfiguration, InteractionMatrixConfiguration, \
    InteractionVersusAllConfiguration


def visualize(model, X: pd.DataFrame, n: int = None):
    ovo = calculate_h_stat(model, X, n)
    ova = calculate_h_stat(model, X, n, ONE_VS_ALL)

    fig = plt.figure(figsize=(18, 12))

    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 1:3])

    ovo_heatmap(ovo, ax1, InteractionMatrixConfiguration.default())
    ovo_interaction_graph(ovo, ax2, InteractionGraphConfiguration.default())
    ova_barchart(ova, ax3, InteractionVersusAllConfiguration.default())

    fig.suptitle("H-statistic summary")


def ovo_heatmap(ovo: pd.DataFrame, ax, config: InteractionMatrixConfiguration):
    ovo_copy = ovo.copy()
    ovo_copy['Feature 1'], ovo_copy['Feature 2'] = ovo_copy['Feature 2'], ovo_copy['Feature 1']
    ovo_all_pairs = pd.concat([ovo, ovo_copy])

    ax.set_title(config.TITLE)
    sns.heatmap(ovo_all_pairs.pivot_table("H-statistic", "Feature 1", "Feature 2"), cmap=config.COLOR_MAP, annot=True,
                ax=ax)


def ovo_interaction_graph(ovo: pd.DataFrame, ax, config: InteractionGraphConfiguration):
    G = nx.from_pandas_edgelist(ovo, source="Feature 1", target="Feature 2", edge_attr="H-statistic")
    pos = nx.circular_layout(G)
    nx.draw(G,
            pos,
            ax=ax,
            width=_edge_widths(G, config),
            with_labels=True,
            node_size=config.NODE_SIZE,
            font_size=config.FONT_SIZE,
            font_weight=config.FONT_WEIGHT,
            font_color=config.FONT_COLOR,
            node_color=config.NODE_COLOR,
            edge_color=config.EDGE_COLOR)

    nx.draw_networkx_edge_labels(G,
                                 pos,
                                 ax=ax,
                                 edge_labels=_edge_labels(config, ovo),
                                 font_color=config.FONT_COLOR,
                                 font_weight=config.FONT_WEIGHT
                                 )

    ax.set_title(config.TITLE)


def ova_barchart(ova: pd.DataFrame, ax, config: InteractionVersusAllConfiguration):
    ova.head(config.N_HIGHEST).plot.bar(x="Feature", y="H-statistic", ylabel="H-statistic", cmap="crest",
                                        title=f"Top {config.N_HIGHEST} features with highest h-statistic", ax=ax)

    ax.set_title(config.TITLE)


def _edge_widths(G, config):
    return [elem * config.MAX_EDGE_WIDTH for elem in nx.get_edge_attributes(G, 'H-statistic').values()]


def _edge_labels(config, ovo):
    return {(row['Feature 1'], row['Feature 2']): round(row['H-statistic'], 2) for index, row in
            ovo.head(config.N_HIGHEST_WITH_LABELS).iterrows()}
