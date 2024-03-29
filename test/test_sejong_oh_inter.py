import unittest
from parameterized import parameterized_class

from artemis._utilities.domain import InteractionMethod
from artemis.interactions_methods.model_agnostic import SejongOhMethod
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from .util import california_housing_random_forest, SAMPLE_SIZE, N_REPEAT, CALIFORNIA_SUBSET, has_decreasing_order, wine_random_forest, WINE_SUBSET


MODEL_REG, X_REG, Y_REG = california_housing_random_forest()
MODEL_CLS, X_CLS, Y_CLS = wine_random_forest()


@parameterized_class([
    {
        "model": MODEL_REG,
        "X": X_REG,
        "y": Y_REG,
        "SUBSET": CALIFORNIA_SUBSET
    },
    {
        "model": MODEL_CLS,
        "X": X_CLS,
        "y": Y_CLS,
        "SUBSET": WINE_SUBSET
    },
])
class SejongOhMethodTestCase(unittest.TestCase):
    model = None
    X = None
    y = None
    SUBSET = None

    def test_all_features_sampled(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, n_repeat=N_REPEAT)

        # then

        # expected columns
        self.assertSetEqual(set(sejong_oh_inter.ovo.columns), {"Feature 1", "Feature 2", InteractionMethod.PERFORMANCE_BASED})

        # sample size taken into account
        self.assertEqual(len(sejong_oh_inter.X_sampled), SAMPLE_SIZE)

        # feature importance calculated
        self.assertIsNotNone(sejong_oh_inter.feature_importance)

    def test_subset_of_features_sampled(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, features=self.SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(sejong_oh_inter.ovo), 6)
        self.assertEqual(sejong_oh_inter.features_included, self.SUBSET)

        # sample size taken into account
        self.assertEqual(len(sejong_oh_inter.X_sampled), SAMPLE_SIZE)

    def test_decreasing_order(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE)

        # then
        ovo_vals = list(sejong_oh_inter.ovo[InteractionMethod.PERFORMANCE_BASED].abs())

        # ovo have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))

    def test_plot(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, features=self.SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(InteractionMethod.PERFORMANCE_BASED).accepted_visualizations
        for vis in accepted_vis:
            sejong_oh_inter.plot(vis, show=False)

        # then
        # nothing crashes!


if __name__ == '__main__':
    unittest.main()
