import unittest
from parameterized import parameterized_class

from artemis.interactions_methods.model_specific import ConditionalMinimalDepthMethod
from artemis._utilities.domain import InteractionMethod
from artemis.visualizer._configuration import VisualizationConfigurationProvider
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
class ConditionalMinimalDepthTestCase(unittest.TestCase):
    model = None
    X = None

    def test_ovo_all_features(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model)

        # then
        self.assertSetEqual(set(cond_min.ovo.columns), {"root_variable", "variable", "n_occurences", cond_min.method})
        p = len(self.X.columns)
        self.assertEqual(len(cond_min.ovo), p * p - p)

    def test_plot(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model)

        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(
            InteractionMethod.CONDITIONAL_MINIMAL_DEPTH).accepted_visualizations
        for vis in accepted_vis:
            cond_min.plot(vis, show=False)

        # then
        # nothing crashes!

    def test_minimal_depth_feature_importance(self):
        # when
        cond_min = ConditionalMinimalDepthMethod()
        cond_min.fit(self.model)

        # then
        self.assertSetEqual(set(cond_min.feature_importance["Feature"]), set(self.X.columns))


if __name__ == '__main__':
    unittest.main()
