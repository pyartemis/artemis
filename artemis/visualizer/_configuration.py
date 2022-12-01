from dataclasses import dataclass
from typing import List

from artemis.utilities.domain import InteractionMethod, VisualisationType
from artemis.utilities.exceptions import MethodNotSupportedException


@dataclass
class InteractionGraphConfiguration:
    MAX_EDGE_WIDTH: int = 20
    N_HIGHEST_WITH_LABELS: int = 5
    FONT_COLOR: str = "#3B1F2B"
    FONT_WEIGHT: str = "bold"
    FONT_SIZE: int = 10
    EDGE_COLOR: str = "#FFA500"
    EDGE_COLOR_POS: str = "#24E9D0"
    EDGE_COLOR_NEG: str = "#DB162F"
    NODE_COLOR: str = "#DBDFAC"
    NODE_SIZE: int = 1800
    TITLE: str = "Interaction graph"
    MIN_RELEVANT_INTERACTION: float = 0.05


@dataclass
class InteractionMatrixConfiguration:
    TITLE: str = "Interaction matrix"
    INTERACTION_COLOR_MAP: str = "Purples"
    IMPORTANCE_COLOR_MAP: str = "Greens"


@dataclass
class InteractionVersusAllConfiguration:
    TITLE: str = "Interaction with all other features"
    N_HIGHEST: int = 5

@dataclass
class InteractionVersusOneConfiguration:
    TITLE: str = "Pair interactions"
    N_HIGHEST: int = 10

class VisualisationConfigurationProvider:
    accepted_visualisations = {
        InteractionMethod.H_STATISTIC: [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH,
                                        VisualisationType.BAR_CHART_OVA, VisualisationType.BAR_CHART_OVO,
                                        VisualisationType.HEATMAP],
        InteractionMethod.PERFORMANCE_BASED: [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH,
                                              VisualisationType.BAR_CHART_OVO, VisualisationType.HEATMAP],
        InteractionMethod.VARIABLE_INTERACTION: [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH,
                                                VisualisationType.BAR_CHART_OVO, VisualisationType.HEATMAP],
        InteractionMethod.CONDITIONAL_MINIMAL_DEPTH: [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH,
                                                      VisualisationType.BAR_CHART_OVO, VisualisationType.HEATMAP],
        InteractionMethod.SPLIT_SCORE: [VisualisationType.SUMMARY, VisualisationType.INTERACTION_GRAPH,
                                                VisualisationType.BAR_CHART_OVO, VisualisationType.HEATMAP]
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
        return VisualisationConfiguration(
            accepted_visualisations=cls.accepted_visualisations[InteractionMethod.H_STATISTIC])

    @classmethod
    def _var_inter_config(cls):
        return VisualisationConfiguration(
            accepted_visualisations=cls.accepted_visualisations[InteractionMethod.VARIABLE_INTERACTION])

    @classmethod
    def _perf_based_config(cls):
        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.1

        return VisualisationConfiguration(
            accepted_visualisations=cls.accepted_visualisations[InteractionMethod.PERFORMANCE_BASED],
            interaction_graph=graph_config)

    @classmethod
    def _split_score_config(cls):
        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.1

        return VisualisationConfiguration(
            accepted_visualisations=cls.accepted_visualisations[InteractionMethod.SPLIT_SCORE],
            interaction_graph=graph_config)

    @classmethod
    def _cond_depth_config(cls):

        graph_config = InteractionGraphConfiguration()
        graph_config.MIN_RELEVANT_INTERACTION = 0.6
        graph_config.MAX_EDGE_WIDTH = 3

        return VisualisationConfiguration(
            accepted_visualisations=cls.accepted_visualisations[InteractionMethod.CONDITIONAL_MINIMAL_DEPTH],
            interaction_graph=graph_config
        )


@dataclass
class VisualisationConfiguration:
    accepted_visualisations: List[str]
    interaction_graph: InteractionGraphConfiguration = InteractionGraphConfiguration()
    interaction_matrix: InteractionMatrixConfiguration = InteractionMatrixConfiguration()
    interaction_bar_chart_ova: InteractionVersusAllConfiguration = InteractionVersusAllConfiguration()
    interaction_bar_chart_ovo: InteractionVersusOneConfiguration = InteractionVersusOneConfiguration()
