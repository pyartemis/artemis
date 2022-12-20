import dataclasses

import pandas as pd
from matplotlib import pyplot as plt

from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis._utilities.domain import CorrelationMethod
from artemis._utilities.exceptions import MethodNotFittedException
from artemis._utilities.ops import point_left_side_circle
from artemis.visualizer._configuration import InteractionGraphConfiguration


class FeatureInteractionMethodComparator:
    """
        Feature Interaction Method Comparator.
        It is used for statistical comparison of two different feature interaction methods.
        Calculates Pearson, Kendall and Spearman rank-correlation and plots one vs one profiles of two methods against
        each other. Monotonicity of the plot suggest cohesion in results. Both provided methods must be in fitted state.


        Attributes
        ----------
        ovo_profiles_comparison_plot : Figure
            Matplotlib figure of comparison plots.
        correlations_df : pd.DataFrame
            Pearson, Kendall and Spearman rank correlation values

        References
        ----------
        - https://en.wikipedia.org/wiki/Rank_correlation
        """

    def __init__(self):
        """Constructor for FeatureInteractionMethodComparator"""
        self.ovo_profiles_comparison_plot = None
        self.correlations_df = None

    def summary(self,
                method1: FeatureInteractionMethod,
                method2: FeatureInteractionMethod):
        _assert_fitted_ovo(method1, method2)
        """
        Calculates Feature Interaction Method comparison. 
        Used for asserting stability and cohesion of results for a pair of explanation methods. 

        Parameters
        ----------
        method1 : FeatureInteractionMethod
             First method for comparison
        method2 : FeatureInteractionMethod
             Second method for comparison 
             
        Returns
        -------
        None
        """
        self.correlations_df = self.correlations(method1, method2)
        self.ovo_profiles_comparison_plot = self.comparison_plot(method1, method2, add_correlation_box=True)

    def correlations(self, method1: FeatureInteractionMethod, method2: FeatureInteractionMethod):
        """
        Calculates Pearson, Kendall and Spearman rank correlation DataFrame.

        Parameters
        ----------
        method1 : FeatureInteractionMethod
             First method for comparison
        method2 : FeatureInteractionMethod
             Second method for comparison

        Returns
        -------
        None
        """
        correlations = list()
        for correlation_method in dataclasses.fields(CorrelationMethod):
            correlation_method_name = correlation_method.default
            correlations.append(
                {
                    "method": correlation_method_name,
                    "value": self.correlation(method1, method2, correlation_method_name)
                })

        return pd.DataFrame.from_records(correlations)

    def comparison_plot(self,
                        method1: FeatureInteractionMethod,
                        method2: FeatureInteractionMethod,
                        n_labels: int = 3,
                        add_correlation_box: bool = False,
                        fig_size: tuple = (8, 6)):
        """
        Creates comparison plot for comparing results of two feature interaction methods. Depending on the parameters
        rank correlation might be included on the plot.

        Parameters
        ----------
        method1 : FeatureInteractionMethod
            First method for comparison
        method2 : FeatureInteractionMethod
            Second method for comparison
        n_labels: int
            Number of pairs of features with the greatest interaction values to show labels of, default = 3
        add_correlation_box: bool
            Flag indicating whether to show rank correlation values on the plot, default = False
        fig_size: tuple[int]
            Matplotlib size of the figure, default = (8, 6)


        Returns
        -------
        Figure
        """
        m1_name, m2_name = method1.method, method2.method
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_axisbelow(True)
        plt.grid(True)
        circle_r = 0.2 * min(max(method1._compare_ovo[m1_name]), max(method2._compare_ovo[m2_name]))

        x, y = list(), list()
        for index, row in method1._compare_ovo.iterrows():

            f1, f2 = row["Feature 1"], row["Feature 2"]
            x_curr, y_curr = row[method1.method], method2.interaction_value(f1, f2)
            x.append(x_curr)
            y.append(y_curr)

            if index < n_labels:
                _add_arrow(ax, circle_r, f1, f2, x_curr, y_curr)

        ax.scatter(x, y, color=InteractionGraphConfiguration.NODE_COLOR)

        if method1.interactions_ascending_order:
            plt.gca().invert_xaxis()
        if method2.interactions_ascending_order:
            plt.gca().invert_yaxis()

        if add_correlation_box:

            corr = self.correlations_df
            if self.correlations_df is None:
                corr = self.correlations(method1, method2)

            _add_correlation_box(ax, corr)

        _title_x_y(ax, m1_name, m2_name)

        return fig, ax

    @staticmethod
    def correlation(
            method1: FeatureInteractionMethod,
            method2: FeatureInteractionMethod,
            correlation_method: str = CorrelationMethod.KENDALL):
        """
        Calculates rank correlation of one vs one profiles using a given correlation method.

        Parameters
        ----------
        method1 : FeatureInteractionMethod
             First method for comparison
        method2 : FeatureInteractionMethod
             Second method for comparison
        correlation_method: str
            Correlation method to use, accepted values are ['pearson', 'kendall', 'spearman'], default = 'kendall'

        Returns
        -------
        value of the correlation
        """

        rank = _rank_interaction_values_encoded(method1, method2)

        return rank.corr(method=correlation_method).iloc[0, 1]


def _rank_interaction_values_encoded(method1, method2):
    rank_features_m1 = method1._compare_ovo.apply(lambda row: _alphabetical_order_pair(row), axis=1)
    rank_features_m2 = method2._compare_ovo.apply(lambda row: _alphabetical_order_pair(row), axis=1)
    rank_features_encoded = pd.concat(
        [rank_features_m1.astype('category').cat.codes, rank_features_m2.astype('category').cat.codes], axis=1)

    return rank_features_encoded


def _title_x_y(ax, m1_name, m2_name):
    ax.set_xlabel(m1_name)
    ax.set_ylabel(m2_name)
    ax.set_title(f"{m1_name}\nand\n{m2_name}\nComparison")


def _add_correlation_box(ax, correlations):
    lines = [
        f"{m.default.capitalize()}={round(correlations[correlations['method'] == m.default]['value'].values[0], 3)}"
        for m in dataclasses.fields(CorrelationMethod)
    ]
    lines.insert(0, "Feature pairs rank correlation")
    correlation_box_text = '\n'.join(lines)

    props = dict(boxstyle='round', alpha=0.5, color=InteractionGraphConfiguration.EDGE_COLOR)
    ax.text(0.95, 0.05,
            correlation_box_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment="right", bbox=props)


def _add_arrow(ax, circle_r, f1, f2, v1, v2):
    ax.annotate("-".join([f1, f2]),
                xy=(v1, v2),
                xycoords='data',
                xytext=point_left_side_circle(v1, v2, circle_r),
                textcoords='data',
                size=8,
                bbox=dict(boxstyle="round", alpha=0.1, color=InteractionGraphConfiguration.EDGE_COLOR),
                arrowprops=dict(
                    arrowstyle="simple",
                    fc="0.6",
                    connectionstyle="arc3",
                    color=InteractionGraphConfiguration.EDGE_COLOR))


def _assert_fitted_ovo(method1: FeatureInteractionMethod, method2: FeatureInteractionMethod):
    if not _suitable_for_ovo(method1):
        raise MethodNotFittedException(method1.method)

    if not _suitable_for_ovo(method2):
        raise MethodNotFittedException(method2.method)


def _suitable_for_ovo(method: FeatureInteractionMethod):
    return method._compare_ovo is not None


def _alphabetical_order_pair(row):
    features_alphabetical = sorted([row["Feature 1"], row["Feature 2"]])

    return "-".join(features_alphabetical)
