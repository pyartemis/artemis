from abc import abstractmethod
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis._utilities.domain import ProgressInfoLog, VisualizationType
from artemis._utilities.ops import get_predict_function, sample_if_not_none, all_if_none
from artemis._utilities.pd_calculator import PartialDependenceCalculator
from artemis._utilities.zenplot import (
    get_pd_dict,
    get_pd_pairs_values,
    get_second_feature,
    find_next_pair_index,
)


class PartialDependenceBasedMethod(FeatureInteractionMethod):
    def __init__(self, method: str, random_state: Optional[int] = None):
        super().__init__(method, random_state=random_state)
        self.pd_calculator = None

    @property
    def _interactions_ascending_order(self):
        return False

    def plot(
        self,
        vis_type: str = VisualizationType.HEATMAP,
        title: str = "default",
        figsize: Tuple[float, float] = (8, 6),
        show: bool = True,
        **kwargs
    ):
        super().plot(vis_type, title, figsize, show, **kwargs)

    def fit(
        self,
        model,
        X: pd.DataFrame,
        n: Optional[int] = None,
        predict_function: Optional[Callable] = None,
        features: Optional[List[str]] = None,
        show_progress: bool = False,
        batchsize: int = 2000,
        pd_calculator: Optional[PartialDependenceCalculator] = None,
    ):
        """Calculates Partial Dependence Based Feature Interactions Strength and Feature Importance for the given model.

        Parameters
        ----------
        model : object
            Model to be explained, should have predict_proba or predict method, or predict_function should be provided.
        X : pd.DataFrame
            Data used to calculate interactions. If n is not None, n rows from X will be sampled.
        n : int, optional
            Number of samples to be used for calculation of interactions. If None, all rows from X will be used. Default is None.
        predict_function : Callable, optional
            Function used to predict model output. It should take model and dataset and outputs predictions.
            If None, `predict_proba` method will be used if it exists, otherwise `predict` method. Default is None.
        features : List[str], optional
            List of features for which interactions will be calculated. If None, all features from X will be used. Default is None.
        show_progress : bool
            If True, progress bar will be shown. Default is False.
        batchsize : int
            Batch size for calculating partial dependence. Prediction requests are collected until the batchsize is exceeded,
            then the model is queried for predictions jointly for many observations. It speeds up the operation of the method.
            Default is 2000.
        pd_calculator : PartialDependenceCalculator, optional
            PartialDependenceCalculator object containing partial dependence values for a given model and dataset.
            Providing this object speeds up the calculation as partial dependence values do not need to be recalculated.
            If None, it will be created from scratch. Default is None.
        """
        self.predict_function = get_predict_function(model, predict_function)
        self.model = model

        self.X_sampled = sample_if_not_none(self._random_generator, X, n)
        self.features_included = all_if_none(X.columns, features)
        self.pairs = list(combinations(self.features_included, 2))

        if pd_calculator is None:
            self.pd_calculator = PartialDependenceCalculator(
                self.model, self.X_sampled, self.predict_function, batchsize
            )
        else:
            if pd_calculator.model != self.model:
                raise ValueError(
                    "Model in PDP calculator is different than the model in the method."
                )
            if not pd_calculator.X.equals(self.X_sampled):
                raise ValueError(
                    "Data in PDP calculator is different than the data in the method."
                )
            self.pd_calculator = pd_calculator

        self.ovo = self._calculate_ovo_interactions_from_pd(show_progress=show_progress)

        self._feature_importance_obj = PartialDependenceBasedImportance()
        self.feature_importance = self._feature_importance_obj.importance(
            self.model,
            self.X_sampled,
            features=self.features_included,
            show_progress=show_progress,
            pd_calculator=self.pd_calculator,
        )

    def plot_profile(
        self,
        feature1: str,
        feature2: Optional[str] = None,
        kind: str = "colormesh",
        cmap: str = "RdYlBu_r",
        figsize: tuple = (6, 4),
        show: bool = True,
        path: Optional[str] = None,
    ):
        plt.figure(figsize=figsize)
        if feature2 is not None:
            pair_key = self.pd_calculator._get_pair_key((feature1, feature2))
            pair = self.pd_calculator.pd_pairs[pair_key]

            if kind == "contour":
                cs = plt.contour(
                    pair["f2_values"],
                    pair["f1_values"],
                    pair["pd_values"],
                    colors="black",
                    linewidths=0.5,
                )
                cs2 = plt.contourf(
                    pair["f2_values"], pair["f1_values"], pair["pd_values"], cmap=cmap
                )
                plt.clabel(cs, colors="black")
                clb = plt.colorbar(cs2)
            elif kind == "colormesh":
                cs = plt.pcolormesh(
                    pair["f2_values"],
                    pair["f1_values"],
                    pair["pd_values"],
                    linewidths=0.5,
                    cmap=cmap,
                )
                clb = plt.colorbar()
            clb.ax.set_title("PD value")
            plt.xlabel(pair_key[1])
            plt.ylabel(pair_key[0])
            sns.rugplot(
                self.pd_calculator.X, y=pair_key[0], x=pair_key[1], color="black"
            )
        else:
            single = self.pd_calculator.pd_single[feature1]
            plt.plot(single["f_values"], single["pd_values"])
            plt.xlabel(feature1)
            plt.ylabel("PD value")
            sns.rugplot(self.pd_calculator.X, x=feature1, color="black")
        if not show:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        

    def plot_zenplot(
        self,
        zenpath_length: int = 7,
        kind: str = "colormesh",
        cmap: str = "RdYlBu_r",
        figsize: tuple = (14, 12),
        show: bool = True,
        path: Optional[str] = None,
    ):
        fig = plt.figure(figsize=figsize)
        to_vis = self.ovo.copy().iloc[: (zenpath_length + 1)]
        min_pd, max_pd = get_pd_dict(self.pd_calculator, to_vis)
        pair = to_vis.iloc[0]["Feature 1"], to_vis.iloc[0]["Feature 2"]
        to_vis = to_vis.drop(0)

        id_row, id_col = 0, 0
        nrows, ncols = int(np.floor((zenpath_length + 1) / 2)), int(
            np.floor(zenpath_length / 2) + 1
        )
        continued = False

        for i in range(zenpath_length):
            pair_values = get_pd_pairs_values(self, pair)
            ax = plt.subplot2grid((nrows, ncols), (id_row, id_col), rowspan=1)

            if id_col > id_row:
                if kind == "colormesh":
                    cs = ax.pcolormesh(
                        pair_values["f2_values"],
                        pair_values["f1_values"],
                        pair_values["pd_values"],
                        vmin=min_pd,
                        vmax=max_pd,
                        cmap=cmap,
                    )
                elif kind == "contour":
                    plt.contour(
                        pair_values["f2_values"],
                        pair_values["f1_values"],
                        pair_values["pd_values"],
                        vmin=min_pd,
                        vmax=max_pd,
                        colors="black",
                        linewidths=0.5,
                    )
                    cs = plt.contourf(
                        pair_values["f2_values"],
                        pair_values["f1_values"],
                        pair_values["pd_values"],
                        vmin=min_pd,
                        vmax=max_pd,
                        cmap=cmap,
                    )
                ax.set_ylabel(pair[0])
                id_row += 1
            else:
                if kind == "colormesh":
                    cs = ax.pcolormesh(
                        pair_values["f1_values"],
                        pair_values["f2_values"],
                        pair_values["pd_values"].T,
                        vmin=min_pd,
                        vmax=max_pd,
                        cmap=cmap,
                    )
                elif kind == "contour":
                    plt.contour(
                        pair_values["f1_values"],
                        pair_values["f2_values"],
                        pair_values["pd_values"].T,
                        vmin=min_pd,
                        vmax=max_pd,
                        colors="black",
                        linewidths=0.5,
                    )
                    cs = plt.contourf(
                        pair_values["f1_values"],
                        pair_values["f2_values"],
                        pair_values["pd_values"].T,
                        vmin=min_pd,
                        vmax=max_pd,
                        cmap=cmap,
                    )
                ax.set_title(pair[0], size=10)
                id_col += 1

            idx, continued = find_next_pair_index(to_vis, pair[1])
            if continued:
                pair = pair[1], get_second_feature(pair[1], to_vis.loc[idx])
                if zenpath_length-1 == i:
                    if id_col > id_row:
                        ax.set_ylabel(pair[1])
                    else:
                        ax.set_title(pair[1], size=10)
            else:
                if id_col > id_row:
                    ax.set_ylabel(pair[1])
                else:
                    ax.set_title(pair[1], size=10)
                pair = to_vis.loc[idx]["Feature 1"], to_vis.loc[idx]["Feature 2"]

            to_vis = to_vis.drop(idx)

            plt.tight_layout()
        cbar_ax = fig.add_axes([1, 0.25, 0.05, 0.5])
        clb = plt.colorbar(cs, cax=cbar_ax)
        clb.ax.set_title("PD value")
        if not show:
            plt.savefig(path, dpi=300, bbox_inches='tight')

    @abstractmethod
    def _calculate_ovo_interactions_from_pd(self, show_progress: bool):
        ...
