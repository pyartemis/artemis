class VisualisationNotSupportedException(Exception):
    def __init__(self, method: str, vis_type: str):
        self.method = method
        self.vis_type = vis_type
        super().__init__(f"Visualisation type: {self.vis_type} is not supported for {self.method}")


class MethodNotSupportedException(Exception):
    def __init__(self, method: str):
        self.method = method
        super().__init__(f"Method: {self.method} is not supported")
