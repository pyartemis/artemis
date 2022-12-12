from dataclasses import dataclass
from typing import List

from artemis.utilities.domain import InteractionMethod, VisualizationType
from artemis.utilities.exceptions import MethodNotSupportedException


@dataclass
class InteractionGraphConfiguration:
    MAX_EDGE_WIDTH: int = 20
    N_HIGHEST_WITH_LABELS: int = 5
    FONT_COLOR: str = "#3B1F2B"
    FONT_WEIGHT: str = "bold"
    FONT_SIZE: int = 10
    EDGE_COLOR: str = "rebeccapurple"
    EDGE_COLOR_POS: str = "#24E9D0"
    EDGE_COLOR_NEG: str = "#DB162F"
    NODE_COLOR: str = "green"
    NODE_SIZE: int = 1800
    TITLE: str = "Interaction graph"
    MIN_RELEVANT_INTERACTION: float = 0.05


@dataclass
class InteractionMatrixConfiguration:
    TITLE: str = "Interaction matrix"
    INTERACTION_COLOR_MAP: str = "Purples"
    INTERACTION_COLOR_MAP_REVERSE: str = "Purples_r"
    IMPORTANCE_COLOR_MAP: str = "Greens"
    IMPORTANCE_COLOR_MAP_REVERSE: str = "Greens_r"
    ANNOT_FMT: str = ".3f"
    LINEWIDTHS: float = 0.5
    LINECOLOR: str = "white"
    CBAR_SHRINK: float = 0.8


@dataclass
class InteractionVersusAllConfiguration:
    TITLE: str = "Interaction with all other features"
    TOP_K: int = 10
    COLOR: str = "mediumpurple"


@dataclass
class InteractionVersusOneConfiguration:
    TITLE: str = "Pair interactions"
    TOP_K: int = 10
    COLOR: str = "mediumpurple"


@dataclass
class LollipopSplitScoreConfiguration:
    TITLE: str = "Lollipop boosting model summary"
    SCALE: str = "linear"
    MAX_TREES: float = 0.2
    LABEL_THRESHOLD: float = 0.1
    COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
    SHAPES = ["o", ",", "v", "^", "<", ">"]
    MAX_DEPTH: int = 1
    LABELS: bool = True


@dataclass
class BarChartConditionalDepthConfiguration:
    TITLE: str = "Random forest model summary"
    COLOR_MAP: str = "Purples"
    TOP_K: int = 15
    COLOR: str = "black"


class VisualizationConfigurationProvider:
    accepted_visualizations = {
        InteractionMethod.H_STATISTIC: [
            VisualizationType.SUMMARY,
            VisualizationType.INTERACTION_GRAPH,
            VisualizationType.BAR_CHART_OVA,
            VisualizationType.BAR_CHART_OVO,
            VisualizationType.HEATMAP,
        ],
        InteractionMethod.PERFORMANCE_BASED: [
            VisualizationType.SUMMARY,
            VisualizationType.INTERACTION_GRAPH,
            VisualizationType.BAR_CHART_OVO,
            VisualizationType.HEATMAP,
        ],
        InteractionMethod.VARIABLE_INTERACTION: [
            VisualizationType.SUMMARY,
            VisualizationType.INTERACTION_GRAPH,
            VisualizationType.BAR_CHART_OVO,
            VisualizationType.HEATMAP,
        ],
        InteractionMethod.CONDITIONAL_MINIMAL_DEPTH: [
            VisualizationType.SUMMARY,
            VisualizationType.INTERACTION_GRAPH,
            VisualizationType.BAR_CHART_OVO,
            VisualizationType.HEATMAP,
            VisualizationType.BAR_CHART_CONDITIONAL,
        ],
        InteractionMethod.SPLIT_SCORE: [
            VisualizationType.SUMMARY,
            VisualizationType.INTERACTION_GRAPH,
            VisualizationType.BAR_CHART_OVO,
            VisualizationType.HEATMAP,
            VisualizationType.LOLLIPOP,
        ],
    }

    @classmethod
    def get(cls, method: str):
        if method == InteractionMethod.H_STATISTIC:
            return cls._h_stat_config()
        elif method == InteractionMethod.VARIABLE_INTERACTION:
            return cls._var_inter_config()
        elif method == InteractionMethod.PERFORMANCE_BASED:
            return cls._perf_based_config()
        elif method == InteractionMethod.SPLIT_SCORE:
            return cls._split_score_config()
        elif method == InteractionMethod.CONDITIONAL_MINIMAL_DEPTH:
            return cls._cond_depth_config()
        else:
            raise MethodNotSupportedException(method)

    @classmethod
    def _h_stat_config(cls):
        return VisualizationConfiguration(
            accepted_visualizations=cls.accepted_visualizations[
                InteractionMethod.H_STATISTIC
            ]
        )

    @classmethod
    def _var_inter_config(cls):
        return VisualizationConfiguration(
            accepted_visualizations=cls.accepted_visualizations[
                InteractionMethod.VARIABLE_INTERACTION
            ]
        )

    @classmethod
    def _perf_based_config(cls):
        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.1

        return VisualizationConfiguration(
            accepted_visualizations=cls.accepted_visualizations[
                InteractionMethod.PERFORMANCE_BASED
            ],
            interaction_graph=graph_config,
        )

    @classmethod
    def _split_score_config(cls):
        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.1

        return VisualizationConfiguration(
            accepted_visualizations=cls.accepted_visualizations[
                InteractionMethod.SPLIT_SCORE
            ],
            interaction_graph=graph_config,
        )

    @classmethod
    def _cond_depth_config(cls):

        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.6
        graph_config.MAX_EDGE_WIDTH = 3

        return VisualizationConfiguration(
            accepted_visualizations=cls.accepted_visualizations[
                InteractionMethod.CONDITIONAL_MINIMAL_DEPTH
            ],
            interaction_graph=graph_config,
        )


@dataclass
class VisualizationConfiguration:
    accepted_visualizations: List[str]
    interaction_graph: InteractionGraphConfiguration = InteractionGraphConfiguration()
    interaction_matrix: InteractionMatrixConfiguration = (
        InteractionMatrixConfiguration()
    )
    interaction_bar_chart_ova: InteractionVersusAllConfiguration = (
        InteractionVersusAllConfiguration()
    )
    interaction_bar_chart_ovo: InteractionVersusOneConfiguration = (
        InteractionVersusOneConfiguration()
    )
    lollipop: LollipopSplitScoreConfiguration = LollipopSplitScoreConfiguration()
    interaction_bar_chart_conditional: BarChartConditionalDepthConfiguration = BarChartConditionalDepthConfiguration()
