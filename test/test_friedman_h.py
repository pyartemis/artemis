import unittest
from parameterized import parameterized_class

from .util import california_housing_random_forest, has_decreasing_order, CALIFORNIA_SUBSET, SAMPLE_SIZE, wine_random_forest, WINE_SUBSET
from artemis.utilities.domain import InteractionMethod
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod
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
class FriedmanHStatisticMethodTestCase(unittest.TestCase):
    model = None
    X = None
    SUBSET = None
    SAMPLE_SIZE = 5

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
        h_stat.fit(self.model, self.X, SAMPLE_SIZE, features=self.SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(h_stat.ova), 4)
        self.assertEqual(len(h_stat.ovo), 6)
        self.assertEqual(h_stat.features_included, self.SUBSET)

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
        h_stat.fit(self.model, self.X, SAMPLE_SIZE, features=self.SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(InteractionMethod.H_STATISTIC).accepted_visualizations
        for vis in accepted_vis:
            h_stat.plot(vis, show=False)

        # then
        # nothing crashes!

if __name__ == '__main__':
    unittest.main()
