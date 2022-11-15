from dataclasses import dataclass


@dataclass
class Method:
    H_STATISTIC: str = "Friedman H-statistic"
    VARIABLE_INTERACTION: str = "Greenwell Variable Interaction"
    PERFORMANCE_BASED: str = "Sejong Oh Performance Based"
    CONDITIONAL_MINIMAL_DEPTH: str = "Conditional Minimal Depth"


@dataclass
class InteractionCalculationStrategy:
    ONE_VS_ONE: str = 'one vs one'
    ONE_VS_ALL: str = 'one vs all'


@dataclass
class VisualisationType:
    SUMMARY: str = "summary"
    INTERACTION_GRAPH: str = "interaction graph"
    BAR_CHART: str = "bar chart"
    HEATMAP: str = "heatmap"


@dataclass
class ProblemType:
    REGRESSION: str = "regression"
    CLASSIFICATION: str = "classification"
