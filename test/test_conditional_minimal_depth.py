import unittest

from artemis.interactions_methods.model_specific import ConditionalMinimalDepthMethod
from artemis.utilities.domain import InteractionMethod
from artemis.visualisation.configuration import VisualisationConfigurationProvider
from test.util import california_housing_random_forest


class ConditionalMinimalDepthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.X, _ = california_housing_random_forest()

    def test_ovo_all_features(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model, self.X)

        # then
        self.assertSetEqual(set(cond_min.ovo.columns), {"root_variable", "variable", "n_occurences", cond_min.method})
        self.assertEqual(len(cond_min.ovo), 8 * 8 - 8)

    def test_plot(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model, self.X)

        # allowed plots are generated without exception
        accepted_vis = VisualisationConfigurationProvider.get(
            InteractionMethod.CONDITIONAL_MINIMAL_DEPTH).accepted_visualisations
        for vis in accepted_vis:
            cond_min.plot(vis)

        # then
        # nothing crashes!

    def test_minimal_depth_variable_importance(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model, self.X)

        # then
        self.assertSetEqual(set(cond_min.variable_importance["Feature"]), set(self.X.columns))


if __name__ == '__main__':
    unittest.main()
