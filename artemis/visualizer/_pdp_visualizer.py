from typing import List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator
import pandas as pd


class PartialDependenceVisualizer:
    """
    Visualizer of 1-dimensianal and 2-dimensional partial dependence plots. 
    It wraps scikit-learn PartialDependenceDisplay.from_estimator() method, 
    so only models implementing predict functions in scikit-learn API are supported.

    Attributes
    ----------
    model : sklearn.BaseEstimator 
        Model for which partial dependence plot will be generated.
    X : pd.DataFrame
        Data used to calculate partial dependence functions.
    """
    def __init__(self, model: BaseEstimator, X: pd.DataFrame):
        """Constructor for PartialDependenceVisualizer

        Parameters
        ----------
        model : sklearn.BaseEstimator 
            Model for which partial dependence plot will be generated.
        X : pd.DataFrame
            Data used to calculate partial dependence functions.
        """
        self.model = model
        self.X = X

    def plot(
        self,
        features: List[Union[int, str, Tuple[int, int], Tuple[str, str]]],
        grid_resolution: int = 100,
        title: str = "Partial Dependence",
        figsize: tuple = (12, 6),
        show: bool = True,
        ax = None,
        **kwargs
    ):
        """Plot partial dependence plot.

        Parameters
        ----------
        features : int, str, (int, int), or (str, str) 
            Features for which partial dependence plot will be generated. 
            If one feature is provided, 1-dimensional PDP will be returned, two features -- 2-dimensional PDP.
        grid_resolution : int
            The number of equally spaced points on the axes of the plots, for each target feature. Default is 100. 
        title : str 
            Title of plot. Default is 'Partial Dependence'.
        figsize : (float, float) 
            Size of plot. Default is (12, 6).
        show : bool 
            Whether to show plot. Default is True.
        **kwargs : Other Parameters
            Additional parameters for plot. Passed to PartialDependenceDisplay.from_estimator() method.
        """
        fig, ax = plt.subplots(figsize=figsize)
        PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features,
            grid_resolution=grid_resolution,
            _ax=ax,
            **kwargs
        )
        ax.set_title(title)
        if show:
            plt.show()
        else:
            return fig
