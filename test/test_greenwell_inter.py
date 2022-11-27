import unittest

from .util import california_housing_random_forest, has_decreasing_order, CALIFORNIA_SUBSET, SAMPLE_SIZE
from artemis.utilities.domain import InteractionMethod, VisualisationType
from artemis.interactions_methods.model_agnostic import GreenwellMethod
from artemis.utilities.exceptions import VisualisationNotSupportedException
from artemis.visualisation.configuration import VisualisationConfigurationProvider


class GreenwellMethodUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model, self.X, _ = california_housing_random_forest()

    def test_all_features_sampled(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE)

        # then

        # expected columns
        self.assertSetEqual(set(greenwell_inter.ovo.columns), {"Feature 1", "Feature 2", InteractionMethod.VARIABLE_INTERACTION})

        # sample size taken into account
        self.assertEqual(len(greenwell_inter.X_sampled), SAMPLE_SIZE)

        # variable importance calculated
        self.assertIsNotNone(greenwell_inter.variable_importance)

    def test_subset_of_features_sampled(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(greenwell_inter.ovo), 6)
        self.assertEqual(greenwell_inter.features_included, CALIFORNIA_SUBSET)

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
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualisationConfigurationProvider.get(InteractionMethod.VARIABLE_INTERACTION).accepted_visualisations
        for vis in accepted_vis:
            greenwell_inter.plot(vis)

        # then
        # nothing crashes!

    def test_should_raise_VisualisationNotSupportedException(self):
        # when
        greenwell_inter = GreenwellMethod()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # barchart is not supported for greenwell (no OvA), so this should raise VisualisationNotSupportedException
        with self.assertRaises(VisualisationNotSupportedException):
            greenwell_inter.plot(VisualisationType.BAR_CHART_OVA)


if __name__ == '__main__':
    unittest.main()
