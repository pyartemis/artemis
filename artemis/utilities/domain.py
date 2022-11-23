from dataclasses import dataclass


@dataclass
class InteractionMethod:
    H_STATISTIC: str = "Friedman H-statistic"
    VARIABLE_INTERACTION: str = "Greenwell Variable Interaction"
    PERFORMANCE_BASED: str = "Sejong Oh Performance Based"


@dataclass
class ImportanceMethod:
    PERMUTATION_IMPORTANCE: str = "Permutation importance"
    PDP_BASED_IMPORTANCE: str = "Partial dependence based"


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


@dataclass
class ProgressInfoLog:
    CALC_OVO: str = f'Calculating {InteractionCalculationStrategy.ONE_VS_ONE} profile'
    CALC_OVA: str = f'Calculating {InteractionCalculationStrategy.ONE_VS_ALL} profile'
    CALC_VAR_IMP: str = 'Calculating variable importance'
