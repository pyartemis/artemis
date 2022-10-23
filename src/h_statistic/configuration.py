from dataclasses import dataclass


@dataclass
class InteractionGraphConfiguration:
    MAX_EDGE_WIDTH: int = 20
    N_HIGHEST_WITH_LABELS: int = 5
    FONT_COLOR: str = "#3B1F2B"
    FONT_WEIGHT: str = "bold"
    FONT_SIZE: int = 10
    EDGE_COLOR: str = "#DB162F"
    NODE_COLOR: str = "#DBDFAC"
    NODE_SIZE: int = 1500
    TITLE: str = "Interaction graph"

    @staticmethod
    def default():
        return InteractionGraphConfiguration()


@dataclass
class InteractionMatrixConfiguration:
    TITLE: str = "Interaction matrix"
    COLOR_MAP: str = "crest"

    @staticmethod
    def default():
        return InteractionMatrixConfiguration()


@dataclass
class InteractionVersusAllConfiguration:
    TITLE: str = "Interaction with all other features"
    N_HIGHEST: int = 5

    @staticmethod
    def default():
        return InteractionVersusAllConfiguration()
