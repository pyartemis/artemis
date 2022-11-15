from matplotlib import pyplot as plt

from artemis.interactions_methods._method import FeatureInteractionMethod
from artemis.utilities.exceptions import MethodNotFittedException
from artemis.utilities.ops import point_on_circle
from artemis.visualisation.configuration import InteractionGraphConfiguration


class FeatureInteractionMethodComparator:
    def __init__(self):
        self.ovo_profiles_comparison_plot = None
        self.show_labels = 5
        self.arrow_length_const = 0.15

    def compare_methods(self,
                        method1: FeatureInteractionMethod,
                        method2: FeatureInteractionMethod):
        _assert_fitted_ovo(method1, method2)
        self.ovo_profiles_comparison_plot = self.comparison_plot_ovo(method1, method2)

    def comparison_plot_ovo(self, method1, method2):
        m1_name, m2_name = method1.method, method2.method
        fig, ax = plt.subplots(figsize=(12, 8))

        circle_r = self.arrow_length_const * min(max(method1.ovo[m1_name]), max(method2.ovo[m2_name]))

        for index, row in method1.ovo.iterrows():

            f1, f2 = row["Feature 1"], row["Feature 2"]
            v1, v2 = row[method1.method], method2.interaction_value(f1, f2)
            ax.scatter(v1, v2, color=InteractionGraphConfiguration.NODE_COLOR)

            if index < self.show_labels:
                ax.annotate("-".join([f1, f2]),
                            (v1, v2),
                            xytext=point_on_circle(v1, v2, circle_r),
                            size=8,
                            bbox=dict(boxstyle="round", alpha=0.1),
                            arrowprops=dict(
                                arrowstyle="simple",
                                fc="0.6",
                                connectionstyle="arc3, rad=0.3",
                                color=InteractionGraphConfiguration.EDGE_COLOR_POS))

        ax.set_xlabel(m1_name)
        ax.set_ylabel(m2_name)
        ax.set_title(f"{m1_name} and {m2_name} comparison")
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax


def _assert_fitted_ovo(method1: FeatureInteractionMethod, method2: FeatureInteractionMethod):
    if not _suitable_for_ovo(method1):
        raise MethodNotFittedException(method1.method)

    if not _suitable_for_ovo(method2):
        raise MethodNotFittedException(method2.method)


def _suitable_for_ovo(method: FeatureInteractionMethod):
    return method.ovo is not None
