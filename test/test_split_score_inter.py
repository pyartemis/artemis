import unittest

from .util import california_housing_random_forest, california_housing_boosting_models, has_decreasing_order, CALIFORNIA_SUBSET
from artemis.utilities.domain import InteractionMethod, VisualizationType
from artemis.utilities.split_score_metrics import SplitScoreInteractionMetric, SplitScoreImportanceMetric, _LGBM_UNSUPPORTED_METRICS
from artemis.interactions_methods.model_specific import SplitScoreMethod
from artemis.utilities.exceptions import VisualizationNotSupportedException, MetricNotSupportedException, ModelNotSupportedException
from artemis.visualizer._configuration import VisualizationConfigurationProvider
from dataclasses import fields

    
class SplitScoreMethodUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_xgb, self.model_lgbm, self.model_xgb_bis, self.model_lgbm_bis, _, _ = california_housing_boosting_models()
        self.model_rf, _, _ = california_housing_random_forest()

    def test_all_metric_combinations_xgb(self):
        for int_method in fields(SplitScoreInteractionMetric):
            for imp_method in fields(SplitScoreImportanceMetric):
                # when
                inter = SplitScoreMethod()
                inter.fit(self.model_xgb, 
                                interaction_selected_metric = int_method.default, 
                                importance_selected_metric = imp_method.default)

                # then

                # expected columns
                self.assertListEqual(list(inter.ovo.columns), ["Feature 1", "Feature 2", InteractionMethod.SPLIT_SCORE])
                self.assertListEqual(list(inter.variable_importance.columns), ["Feature", "Value"])

                # variable importance calculated
                self.assertIsNotNone(inter.full_ovo)
                self.assertIsNotNone(inter.full_result)

    def test_all_metric_combinations_lgbm(self):
        for int_method in fields(SplitScoreInteractionMetric):
            for imp_method in fields(SplitScoreImportanceMetric):
                # when 
                if int_method.default in _LGBM_UNSUPPORTED_METRICS or imp_method.default in _LGBM_UNSUPPORTED_METRICS:
                    # then expect exception
                    with self.assertRaises(MetricNotSupportedException):
                        inter = SplitScoreMethod()
                        inter.fit(self.model_lgbm, 
                                interaction_selected_metric = int_method.default, 
                                importance_selected_metric = imp_method.default)
                else: 
                    inter = SplitScoreMethod()
                    inter.fit(self.model_lgbm,
                            interaction_selected_metric = int_method.default, 
                            importance_selected_metric = imp_method.default)

                    # expected columns
                    self.assertListEqual(list(inter.ovo.columns), ["Feature 1", "Feature 2", InteractionMethod.SPLIT_SCORE])
                    self.assertListEqual(list(inter.variable_importance.columns), ["Feature", "Value"])

                    # variable importance calculated
                    self.assertIsNotNone(inter.full_ovo)
                    self.assertIsNotNone(inter.full_result)


    def test_decreasing_order(self):
        # when
        inter = SplitScoreMethod()
        inter.fit(self.model_xgb)
        inter2 = SplitScoreMethod()
        inter2.fit(self.model_lgbm)

        # then
        ovo_vals = list(inter.ovo[InteractionMethod.SPLIT_SCORE])
        ovo_vals2 = list(inter2.ovo[InteractionMethod.SPLIT_SCORE])

        # ovo have values sorted in decreasing order
        self.assertTrue(has_decreasing_order(ovo_vals))
        self.assertTrue(has_decreasing_order(ovo_vals2))

    def test_plot(self):
        # when
        inter = SplitScoreMethod()
        inter.fit(self.model_xgb_bis)
        inter2 = SplitScoreMethod()
        inter2.fit(self.model_lgbm_bis)
        # allowed plots are generated without exception
        accepted_vis = VisualizationConfigurationProvider.get(InteractionMethod.SPLIT_SCORE).accepted_visualizations
        for vis in accepted_vis:
            inter.plot(vis, show=False)
            inter2.plot(vis, show=False)
        # then
        # nothing crashes!

    def test_progress_bar(self):
        # when progress bar i shown
        inter = SplitScoreMethod()
        inter.fit(self.model_xgb, show_progress=True)    
        inter.fit(self.model_lgbm, show_progress=True)    
        # then
        # nothing crashes!

    def test_not_only_def_interactions(self):
        # when not only interactions by definition are calculated
        inter = SplitScoreMethod()
        inter.fit(self.model_xgb, only_def_interactions=False)    
        inter.fit(self.model_lgbm, only_def_interactions=False)    
        # then
        # nothing crashes!

    def test_should_raise_VisualizationNotSupportedException(self):
        # when
        inter = SplitScoreMethod()
        inter.fit(self.model_xgb)
        inter2 = SplitScoreMethod()
        inter2.fit(self.model_lgbm)

        # barchart (OvA) is not supported 
        with self.assertRaises(VisualizationNotSupportedException):
            inter.plot(VisualizationType.BAR_CHART_OVA)
        with self.assertRaises(VisualizationNotSupportedException):
            inter2.plot(VisualizationType.BAR_CHART_OVA)

    def test_should_raise_ModelNotSupportedException(self):
        with self.assertRaises(ModelNotSupportedException):
            inter = SplitScoreMethod()
            inter.fit(self.model_rf)

if __name__ == '__main__':
    unittest.main()
