from enum import Enum


class InteractionCalculationStrategy(Enum):
    ONE_VS_ONE = 'one vs one'
    ONE_VS_ALL = 'one vs all'


# hack to allow static imports
ONE_VS_ONE = InteractionCalculationStrategy.ONE_VS_ONE
ONE_VS_ALL = InteractionCalculationStrategy.ONE_VS_ALL
