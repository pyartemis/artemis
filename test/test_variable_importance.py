import unittest
from parameterized import parameterized_class

import pandas as pd
from pandas.testing import assert_frame_equal

from artemis.importance_methods.model_agnostic import PermutationImportance, PartialDependenceBasedImportance
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod
from test.util import toy_input_reg, toy_input_cls


MODEL_REG, X_REG, Y_REG = toy_input_reg()
MODEL_CLS, X_CLS, Y_CLS = toy_input_cls()


@parameterized_class([
    {
        "model": MODEL_REG,
        "X": X_REG,
        "y": Y_REG,
    },
    {
        "model": MODEL_CLS,
        "X": X_CLS,
        "y": Y_CLS,
    },
])
class VariableImportanceUnitTest(unittest.TestCase):
    model = None
    X = None
    y = None

    def test_calculate_permutation_variable_importance(self):
        calculator = PermutationImportance()
        importance = calculator.importance(self.model, self.X, self.y, features=list(self.X.columns))

        self._assert_var_imp_calculated_correctly(importance)

    def test_calculate_pdp_based_variable_importance(self):
        calculator = PartialDependenceBasedImportance()
        importance = calculator.importance(self.model, self.X, features=list(self.X.columns))
        self._assert_var_imp_calculated_correctly(importance)

    def test_use_variable_importance_in_pdp_method(self):
        importance_single = PartialDependenceBasedImportance().importance(self.model, self.X,
                                                                            features=list(self.X.columns))

        h_stat = FriedmanHStatisticMethod()
        h_stat.fit(self.model, self.X)
        importance_h_stat = h_stat.variable_importance

        assert_frame_equal(importance_h_stat, importance_single, rtol=1e-1)  # up to first decimal point

    def _assert_var_imp_calculated_correctly(self, importance):
        self.assertEqual(type(importance), pd.DataFrame)  # resulting type - dataframe
        self.assertSetEqual(set(importance["Feature"]),
                            set(self.X.columns))  # var imp for all features is calculated
        self.assertGreater(importance[importance["Feature"] == "important_feature"]["Importance"].values[0],
                           importance[importance["Feature"] == "noise_feature"]["Importance"].values[0])


if __name__ == '__main__':
    unittest.main()
