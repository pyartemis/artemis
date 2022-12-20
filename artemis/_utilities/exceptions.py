class VisualizationNotSupportedException(Exception):
    def __init__(self, method: str, vis_type: str):
        self.method = method
        self.vis_type = vis_type
        super().__init__(
            f"Visualization type: {self.vis_type} is not supported for {self.method}"
        )


class MethodNotSupportedException(Exception):
    def __init__(self, method: str):
        self.method = method
        super().__init__(f"Method: {self.method} is not supported")


class MethodNotFittedException(Exception):
    def __init__(self, method: str):
        self.method = method
        super().__init__(f"Method: {self.method} is was not fitted. Execute fit() first.")


class ModelNotSupportedException(Exception):
    def __init__(self, package: str, model_class: str):
        self.model_class = model_class
        self.package = package
        super().__init__(
            f"Model of class {self.model_class} from the {self.package} package is not supported"
        )


class MetricNotSupportedException(Exception):
    def __init__(self, package: str, metric: str):
        self.metric = metric
        self.package = package
        super().__init__(f"Metric {self.metric} is not supported for model from the {self.package} package")


class FeatureImportanceWithoutInteractionException(Exception):

    def __init__(self, feature_importance: str, feature_interaction: str):
        self.feature_interaction = feature_interaction
        self.feature_importance = feature_importance
        super().__init__(
            f"Feature importance method {self.feature_importance} can be only " +
            f"calculated together with its {self.feature_interaction} counterpart. "
        )