import random
import unittest

import pandas as pd

from artemis.importance_methods.model_agnostic import PermutationImportance, PartialDependenceBasedImportance
from test.util import toy_input, N


class VariableImportanceUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model, self.X, self.y = toy_input()

    def test_calculate_permutation_variable_importance(self):
        calculator = PermutationImportance()
        importance = calculator.importance(self.model, self.X, self.y, features=list(self.X.columns))

        self._assert_var_imp_calculated_correctly(calculator.method, importance)

    def test_calculate_pdp_based_variable_importance(self):
        calculator = PartialDependenceBasedImportance()
        importance = calculator.importance(self.model, self.X, features=list(self.X.columns))
        self._assert_var_imp_calculated_correctly(calculator.method, importance)

    def test_use_cached_pdp_for_variable_importance(self):
        calculator = PartialDependenceBasedImportance()
        pdp_cache = {
            "important_feature": list(range(N)),
            "noise_feature": [random.random() for _ in range(N)]
        }

        importance = calculator.importance(self.model, self.X, features=list(self.X.columns),
                                           precalculated_pdp=pdp_cache)

        self._assert_var_imp_calculated_correctly(calculator.method, importance)

    def _assert_var_imp_calculated_correctly(self, method, importance):
        self.assertEqual(type(importance), pd.DataFrame)  # resulting type - dataframe
        self.assertSetEqual(set(importance["Feature"]),
                            set(self.X.columns))  # var imp for all features is calculated
        self.assertGreater(importance[importance["Feature"] == "important_feature"][method].values[0],
                           importance[importance["Feature"] == "noise_feature"][method].values[0])


if __name__ == '__main__':
    unittest.main()
