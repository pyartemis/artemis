from typing import List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator
import pandas as pd


class PartialDependenceVisualizer:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame):
        self.model = model
        self.X = X

    def plot(
        self,
        features: List[Union[int, str, Tuple[int, int], Tuple[str, str]]],
        grid_resolution: int = 100,
        title: Optional[str] = "Partial Dependence",
        figsize: tuple = (12, 6),
        show: bool = True,
        ax = None,
        **kwargs
    ):
        fig, ax = plt.subplots(figsize=figsize)
        PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features,
            grid_resolution=grid_resolution,
            ax=ax,
            **kwargs
        )
        ax.set_title(title)
        if show:
            plt.show()
        else:
            return fig
