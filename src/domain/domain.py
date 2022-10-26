from dataclasses import dataclass
from enum import Enum


@dataclass
class Method:
    H_STATISTIC: str = "Friedman H-statistic"
    VARIABLE_INTERACTION: str = "Greenwell Variable Interaction"


class InteractionCalculationStrategy(Enum):
    ONE_VS_ONE = 'one vs one'
    ONE_VS_ALL = 'one vs all'


class VisualisationType:
    SUMMARY = "summary"
    INTERACTION_GRAPH = "interaction graph"
    BAR_CHART = "bar chart"
    HEATMAP = "heatmap"


# hack to allow static imports
ONE_VS_ONE = InteractionCalculationStrategy.ONE_VS_ONE
ONE_VS_ALL = InteractionCalculationStrategy.ONE_VS_ALL

SUMMARY = VisualisationType.SUMMARY
INTERACTION_GRAPH = VisualisationType.INTERACTION_GRAPH
BAR_CHART = VisualisationType.BAR_CHART
HEATMAP = VisualisationType.HEATMAP
