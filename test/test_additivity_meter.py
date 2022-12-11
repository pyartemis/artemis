import unittest

from parameterized import parameterized_class

from artemis.additivity import AdditivityMeter
from test.util import california_housing_random_forest, california_housing_linear_regression

MODEL_REG, X_REG, _ = california_housing_random_forest()
LINEAR_MODEL_REG, _, _ = california_housing_linear_regression()

@parameterized_class([
    {
        "model": MODEL_REG,
        "X": X_REG,
        "linear_model": LINEAR_MODEL_REG
    },
])
class AdditivityMeterUnitTest(unittest.TestCase):
    model = None
    X = None
    linear_model = None

    def setUp(self) -> None:
        X_sample = self.X.sample(n=100)
        self.additivity_meter_rf = AdditivityMeter()
        self.additivity_meter_rf.fit(self.model, X_sample)
        
        self.additivity_meter_linear = AdditivityMeter()
        self.additivity_meter_linear.fit(self.linear_model, X_sample)

    def test_additivity_index_values(self):
        self.assertLessEqual(self.additivity_meter_linear.additivity_index, 1)
        self.assertLessEqual(self.additivity_meter_rf.additivity_index, 1)
        self.assertGreater(self.additivity_meter_linear.additivity_index, self.additivity_meter_rf.additivity_index)
        self.assertEqual(self.additivity_meter_linear.additivity_index, 1)

if __name__ == '__main__':
    unittest.main()
