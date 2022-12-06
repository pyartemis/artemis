import unittest
from .util import california_housing_random_forest, has_decreasing_order, CALIFORNIA_SUBSET, SAMPLE_SIZE
from artemis.utilities.domain import InteractionMethod
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod
from artemis.visualizer._configuration import VisualizationConfigurationProvider


class FriedmanHStatisticMethodTestCase(unittest.TestCase):
    SAMPLE_SIZE = 5

    def setUp(self) -> None:
        self.model, self.X, _ = california_housing_random_forest()

    def test_all_features_sampled(self):
        # when
        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X, SAMPLE_SIZE)

        # then

        # expected columns
        self.assertSetEqual(set(h_stat.ova.columns), {"Feature", InteractionMethod.H_STATISTIC})
        self.assertSetEqual(set(h_stat.ovo.columns), {"Feature 1", "Feature 2", InteractionMethod.H_STATISTIC})

        # ova calculated for all columns
        self.assertSetEqual(set(self.X.columns), set(h_stat.ova["Feature"]))

        # sample size taken into account
        self.assertEqual(len(h_stat.X_sampled), SAMPLE_SIZE)

    def test_subset_of_features_sampled(self):
        # when
        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(h_stat.ova), 4)
        self.assertEqual(len(h_stat.ovo), 6)
        self.assertEqual(h_stat.features_included, CALIFORNIA_SUBSET)

        # sample size taken into account
        self.assertEqual(len(h_stat.X_sampled), SAMPLE_SIZE)

    def test_decreasing_order(self):
        # when
        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X, SAMPLE_SIZE)

        # then
        ovo_vals = list(h_stat.ovo[InteractionMethod.H_STATISTIC])
        ova_vals = list(h_stat.ova[InteractionMethod.H_STATISTIC])

        # both ovo and ova have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))
        self.assertTrue(has_decreasing_order(ova_vals))

    def test_plot(self):
        # when
        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(InteractionMethod.H_STATISTIC).accepted_visualizations
        for vis in accepted_vis:
            h_stat.plot(vis, show=False)

        # then
        # nothing crashes!

    def test_cache_pdp_for_var_imp(self):
        # when
        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # then
        self.assertIsNotNone(h_stat._pdp_cache)
        self.assertIsNotNone(h_stat.variable_importance)


if __name__ == '__main__':
    unittest.main()
