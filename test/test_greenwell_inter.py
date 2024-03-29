import unittest
from parameterized import parameterized_class

from .util import california_housing_random_forest, has_decreasing_order, CALIFORNIA_SUBSET, SAMPLE_SIZE, wine_random_forest, WINE_SUBSET
from artemis._utilities.domain import InteractionMethod, VisualizationType
from artemis.interactions_methods.model_agnostic import GreenwellMethod
from artemis._utilities.exceptions import VisualizationNotSupportedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider

MODEL_REG, X_REG, _ = california_housing_random_forest()
MODEL_CLS, X_CLS, _ = wine_random_forest()


@parameterized_class([
    {
        "model": MODEL_REG,
        "X": X_REG,
        "SUBSET": CALIFORNIA_SUBSET
    },
    {
        "model": MODEL_CLS,
        "X": X_CLS,
        "SUBSET": WINE_SUBSET
    },
])
class GreenwellMethodUnitTest(unittest.TestCase):
    model = None
    X = None
    SUBSET = None

    def test_all_features_sampled(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE)

        # then

        # expected columns
        self.assertSetEqual(set(greenwell_inter.ovo.columns),
                            {"Feature 1", "Feature 2", InteractionMethod.VARIABLE_INTERACTION})

        # sample size taken into account
        self.assertEqual(len(greenwell_inter.X_sampled), SAMPLE_SIZE)

        # feature importance calculated
        self.assertIsNotNone(greenwell_inter.feature_importance)

    def test_subset_of_features_sampled(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=self.SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(greenwell_inter.ovo), 6)
        self.assertEqual(greenwell_inter.features_included, self.SUBSET)

        # sample size taken into account
        self.assertEqual(len(greenwell_inter.X_sampled), SAMPLE_SIZE)

    def test_decreasing_order(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE)

        # then
        ovo_vals = list(greenwell_inter.ovo[InteractionMethod.VARIABLE_INTERACTION])

        # ovo have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))

    def test_plot(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=self.SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(
            InteractionMethod.VARIABLE_INTERACTION).accepted_visualizations
        for vis in accepted_vis:
            greenwell_inter.plot(vis, show=False)

        # then
        # nothing crashes!

    def test_should_raise_VisualizationNotSupportedException(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=self.SUBSET)

        # barchart is not supported for greenwell (no OvA), so this should raise VisualizationNotSupportedException
        with self.assertRaises(VisualizationNotSupportedException):
            greenwell_inter.plot(VisualizationType.BAR_CHART_OVA)


if __name__ == '__main__':
    unittest.main()
