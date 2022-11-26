import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from artemis.importance_methods.model_agnostic import PermutationImportance, PartialDependenceBasedImportance
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod
from test.util import toy_input


class VariableImportanceUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model, self.X, self.y = toy_input()

    def test_calculate_permutation_variable_importance(self):
        calculator = PermutationImportance()
        importance = calculator.importance(self.model, self.X, self.y, features=list(self.X.columns))

        self._assert_var_imp_calculated_correctly(importance)

    def test_calculate_pdp_based_variable_importance(self):
        calculator = PartialDependenceBasedImportance()
        importance = calculator.importance(self.model, self.X, features=list(self.X.columns))
        self._assert_var_imp_calculated_correctly(importance)

    def test_use_cached_pdp_for_variable_importance(self):
        importance_no_cache = PartialDependenceBasedImportance().importance(self.model, self.X,
                                                                            features=list(self.X.columns))

        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X)
        importance_cache = h_stat.variable_importance

        self.assertIsNotNone(h_stat._pdp_cache)  # cache was used
        assert_frame_equal(importance_cache, importance_no_cache, rtol=1e-1)  # up to first decimal point

    def _assert_var_imp_calculated_correctly(self, importance):
        self.assertEqual(type(importance), pd.DataFrame)  # resulting type - dataframe
        self.assertSetEqual(set(importance["Feature"]),
                            set(self.X.columns))  # var imp for all features is calculated
        self.assertGreater(importance[importance["Feature"] == "important_feature"]["Value"].values[0],
                           importance[importance["Feature"] == "noise_feature"]["Value"].values[0])


if __name__ == '__main__':
    unittest.main()
