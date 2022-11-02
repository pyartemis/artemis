import unittest

from util import california_housing_random_forest, has_decreasing_order, CALIFORNIA_SUBSET, SAMPLE_SIZE
from src.domain.domain import Method, VisualisationType
from src.methods.partial_dependence_based.variable_interaction.api import GreenwellVariableInteraction
from src.util.exceptions import VisualisationNotSupportedException
from src.visualisation.configuration import VisualisationConfigurationProvider


class GreenwellVariableInteractionUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model, self.X, _ = california_housing_random_forest()

    def test_failing(self):
        self.assertEqual(False, True)

    def test_all_features_sampled(self):
        # when
        greenwell_inter = GreenwellVariableInteraction()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE)

        # then

        # expected columns
        self.assertSetEqual(set(greenwell_inter.ovo.columns), {"Feature 1", "Feature 2", Method.VARIABLE_INTERACTION})

        # sample size taken into account
        self.assertEqual(len(greenwell_inter.X_sampled), SAMPLE_SIZE)

    def test_subset_of_features_sampled(self):
        # when
        greenwell_inter = GreenwellVariableInteraction()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(greenwell_inter.ovo), 6)
        self.assertEqual(greenwell_inter.features_included, CALIFORNIA_SUBSET)

        # sample size taken into account
        self.assertEqual(len(greenwell_inter.X_sampled), SAMPLE_SIZE)

    def test_decreasing_order(self):
        # when
        greenwell_inter = GreenwellVariableInteraction()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE)

        # then
        ovo_vals = list(greenwell_inter.ovo[Method.VARIABLE_INTERACTION])

        # ovo have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))

    def test_plot(self):
        # when
        greenwell_inter = GreenwellVariableInteraction()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualisationConfigurationProvider.get(Method.VARIABLE_INTERACTION).accepted_visualisations
        for vis in accepted_vis:
            greenwell_inter.plot(vis)

        # then
        # nothing crashes!

    def test_should_raise_VisualisationNotSupportedException(self):
        # when
        greenwell_inter = GreenwellVariableInteraction()
        greenwell_inter.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # barchart is not supported for greenwell (no OvA), so this should raise VisualisationNotSupportedException
        with self.assertRaises(VisualisationNotSupportedException):
            greenwell_inter.plot(VisualisationType.BAR_CHART)


if __name__ == '__main__':
    unittest.main()
