from itertools import combinations
import unittest
import numpy as np
from parameterized import parameterized_class
from artemis.utilities.ops import get_predict_function

from artemis.utilities.pd_calculator import PartialDependenceCalculator

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
class PartialDependenceCalculatorTestCase(unittest.TestCase):
    model = None
    X = None
    SUBSET = None
    SAMPLE_SIZE = 5

    def test_all_features(self):
        X_sampled = self.X.sample(SAMPLE_SIZE)
        # when
        pdp_calc = PartialDependenceCalculator(self.model, X_sampled, get_predict_function(self.model))
        pdp_calc.calculate_pd_pairs()
        pdp_calc.calculate_pd_single()
        pdp_calc.calculate_pd_minus_single()

        # then

        # expected columns
        self.assertSetEqual(set(pdp_calc.pd_single.keys()), set(X_sampled.columns))
        self.assertSetEqual(set(pdp_calc.pd_minus_single.keys()), set(X_sampled.columns))
        self.assertSetEqual(set(pdp_calc.pd_pairs.keys()), set(combinations(X_sampled.columns, 2)))

        # expect non nan values
        for var in X_sampled.columns:
            self.assertFalse(np.isnan(pdp_calc.get_pd_single(var)).any())
            self.assertFalse(np.isnan(pdp_calc.get_pd_minus_single(var)).any())

        for var1, var2 in combinations(X_sampled.columns, 2):
            self.assertFalse(np.isnan(pdp_calc.get_pd_pairs(var1, var2)).any())
        
    def test_subset_of_features(self):
        X_sampled = self.X.sample(SAMPLE_SIZE)
        # when
        pdp_calc = PartialDependenceCalculator(self.model, X_sampled, get_predict_function(self.model))
        pdp_calc.calculate_pd_pairs(feature_pairs = combinations(self.SUBSET, 2))
        pdp_calc.calculate_pd_single(features=self.SUBSET)
        pdp_calc.calculate_pd_minus_single(features=self.SUBSET)

        # then

        # expected columns
        self.assertSetEqual(set(pdp_calc.pd_single.keys()), set(X_sampled.columns))
        self.assertSetEqual(set(pdp_calc.pd_minus_single.keys()), set(X_sampled.columns))
        self.assertSetEqual(set(pdp_calc.pd_pairs.keys()), set(combinations(X_sampled.columns, 2)))

        # expect non nan values in subset
        for var in self.SUBSET:
            self.assertFalse(np.isnan(pdp_calc.get_pd_single(var)).any())
            self.assertFalse(np.isnan(pdp_calc.get_pd_minus_single(var)).any())

        for var1, var2 in combinations(self.SUBSET, 2):
            self.assertFalse(np.isnan(pdp_calc.get_pd_pairs(var1, var2)).any())

        # expect nan values in other features
        for var in X_sampled.columns:
            if var not in self.SUBSET:
                self.assertTrue(np.isnan(pdp_calc.get_pd_single(var)).all())
                self.assertTrue(np.isnan(pdp_calc.get_pd_minus_single(var)).all())
        
        for var1, var2 in combinations(X_sampled.columns, 2):
            if var1 not in self.SUBSET or var2 not in self.SUBSET:
                self.assertTrue(np.isnan(pdp_calc.get_pd_pairs(var1, var2)).all())
        
if __name__ == '__main__':
    unittest.main()
