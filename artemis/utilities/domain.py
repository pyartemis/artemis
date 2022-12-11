from dataclasses import dataclass

@dataclass
class InteractionMethod:
    H_STATISTIC: str = "Friedman H-statistic Interaction Measure"
    VARIABLE_INTERACTION: str = "Greenwell Variable Interaction Measure"
    PERFORMANCE_BASED: str = "Sejong Oh Performance Based Interaction Measure"
    SPLIT_SCORE: str = "Split Score Interaction Measure"
    CONDITIONAL_MINIMAL_DEPTH: str = "Conditional Minimal Depth Measure"


@dataclass
class ImportanceMethod:
    PERMUTATION_IMPORTANCE: str = "Permutation Importance"
    PDP_BASED_IMPORTANCE: str = "Partial Dependence Based Importance"
    SPLIT_SCORE_IMPORTANCE: str = "Split Score Based Importance"
    MINIMAL_DEPTH_IMPORTANCE: str = "Minimal Depth Based Importance"


@dataclass
class InteractionCalculationStrategy:
    ONE_VS_ONE: str = "one vs one"
    ONE_VS_ALL: str = "one vs all"


@dataclass
class VisualizationType:
    SUMMARY: str = "summary"
    INTERACTION_GRAPH: str = "graph"
    BAR_CHART_OVA: str = "bar_chart_ova"
    BAR_CHART_OVO: str = "bar_chart"
    HEATMAP: str = "heatmap"
    LOLLIPOP: str = "lollipop"
    BAR_CHART_CONDITIONAL: str = "bar_chart_conditional"


@dataclass
class ProblemType:
    REGRESSION: str = "regression"
    CLASSIFICATION: str = "classification"


@dataclass
class ProgressInfoLog:
    CALC_OVO: str = (
        f"Calculating {InteractionCalculationStrategy.ONE_VS_ONE} interactions"
    )
    CALC_OVA: str = (
        f"Calculating {InteractionCalculationStrategy.ONE_VS_ALL} interactions"
    )
    CALC_VAR_IMP: str = "Calculating feature importance"
    CALC_ADD: str = "Calculating additivity index of the model"


@dataclass
class CorrelationMethod:
    PEARSON: str = "pearson"
    KENDALL: str = "kendall"
    SPEARMAN: str = "spearman"

