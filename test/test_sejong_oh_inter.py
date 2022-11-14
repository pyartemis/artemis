import unittest

from artemis.utilities.domain import Method
from artemis.interactions_methods.model_agnostic_methods import SejongOhMethod
from artemis.visualisation.configuration import VisualisationConfigurationProvider
from .util import california_housing_random_forest, SAMPLE_SIZE, N_REPEAT, CALIFORNIA_SUBSET, has_decreasing_order


class SejongOhMethodTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.X, self.y = california_housing_random_forest()

    def test_all_features_sampled(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, n_repeat=N_REPEAT)

        # then

        # expected columns
        self.assertSetEqual(set(sejong_oh_inter.ovo.columns), {"Feature 1", "Feature 2", Method.PERFORMANCE_BASED})

        # sample size taken into account
        self.assertEqual(len(sejong_oh_inter.X_sampled), SAMPLE_SIZE)

    def test_subset_of_features_sampled(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # then

        # features parameter taken into account
        self.assertEqual(len(sejong_oh_inter.ovo), 6)
        self.assertEqual(sejong_oh_inter.features_included, CALIFORNIA_SUBSET)

        # sample size taken into account
        self.assertEqual(len(sejong_oh_inter.X_sampled), SAMPLE_SIZE)

    def test_decreasing_order(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE)

        # then
        ovo_vals = list(sejong_oh_inter.ovo[Method.PERFORMANCE_BASED])

        # ovo have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))

    def test_plot(self):
        # when
        sejong_oh_inter = SejongOhMethod()
        sejong_oh_inter.fit(self.model, self.X, self.y, SAMPLE_SIZE, features=CALIFORNIA_SUBSET)

        # allowed plots are generated without exception
        accepted_vis = VisualisationConfigurationProvider.get(Method.PERFORMANCE_BASED).accepted_visualisations
        for vis in accepted_vis:
            sejong_oh_inter.plot(vis)

        # then
        # nothing crashes!


if __name__ == '__main__':
    unittest.main()
