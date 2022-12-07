import unittest
from parameterized import parameterized_class

from artemis.comparision import FeatureInteractionMethodComparator
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod, GreenwellMethod
from artemis.utilities.exceptions import MethodNotFittedException
from test.util import california_housing_random_forest, wine_random_forest


MODEL_REG, X_REG, _ = california_housing_random_forest()
MODEL_CLS, X_CLS, _ = wine_random_forest()


@parameterized_class([
    {
        "model": MODEL_REG,
        "X": X_REG
    },
    {
        "model": MODEL_CLS,
        "X": X_CLS
    },
])
class MethodComparatorUnitTest(unittest.TestCase):
    model = None
    X = None

    def setUp(self) -> None:
        self.method_1, self.method_2 = FriedmanHStatisticMethod(), GreenwellMethod()

        self.method_1_fitted, self.method_2_fitted = FriedmanHStatisticMethod(), GreenwellMethod()

        self.method_1_fitted.fit(self.model, self.X, n=10)
        self.method_2_fitted.fit(self.model, self.X, n=10)

    def test_method_not_fitted_exception(self):
        comparator = FeatureInteractionMethodComparator()

        with self.assertRaises(MethodNotFittedException):
            comparator.summary(self.method_1, self.method_2_fitted)

        with self.assertRaises(MethodNotFittedException):
            comparator.summary(self.method_1_fitted, self.method_2)

    def test_should_calculate_correlations(self):
        comparator = FeatureInteractionMethodComparator()

        correlations = comparator.correlations(self.method_1_fitted, self.method_2_fitted)

        self.assertSetEqual(set(correlations["method"]), {"pearson", "kendall", "spearman"})

    def test_should_calculate_comparison_plot(self):
        comparator = FeatureInteractionMethodComparator()

        fig, ax = comparator.comparison_plot(self.method_1_fitted, self.method_2_fitted)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


if __name__ == '__main__':
    unittest.main()
