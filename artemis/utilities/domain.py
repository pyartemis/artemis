from dataclasses import dataclass

from artemis.utilities.exceptions import MethodNotSupportedException


@dataclass
class InteractionMethod:
    H_STATISTIC: str = "Friedman H-statistic"
    VARIABLE_INTERACTION: str = "Greenwell Variable Interaction"
    PERFORMANCE_BASED: str = "Sejong Oh Performance Based"

    @staticmethod
    def get_corresponding_importance(interaction_method: str):

        if interaction_method in [InteractionMethod.H_STATISTIC, InteractionMethod.VARIABLE_INTERACTION]:
            return ImportanceMethod.PDP_BASED_IMPORTANCE
        elif interaction_method == InteractionMethod.PERFORMANCE_BASED:
            return ImportanceMethod.PERMUTATION_IMPORTANCE
        else:
            raise MethodNotSupportedException(interaction_method)


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
