from src.domain.domain import VisualisationType


class VisualisationNotSupportedException(Exception):
    def __init__(self, method: str, vis_type: VisualisationType):
        self.method = method
        self.vis_type = vis_type
        super().__init__(f"Visualisation type: {self.vis_type} is not supported for {self.method}")
