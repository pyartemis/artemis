from dataclasses import dataclass

@dataclass
class SplitScoreInteractionMetric:
    SUM_GAIN: str = "sum_gain"
    SUM_COVER: str = "sum_cover"
    MEAN_GAIN: str = "mean_gain"
    MEAN_COVER: str = "mean_cover"
    MEAN_DEPTH: str = "mean_depth"

@dataclass
class SplitScoreImportanceMetric(SplitScoreInteractionMetric):
    MEAN_WEIGHTED_DEPTH: str = "mean_weighted_depth"
    ROOT_FREQUENCY: str = "root_frequency"
    WEIGHTED_ROOT_FREQUENCY: str = "weighted_root_frequency"
