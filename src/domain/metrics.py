from abc import abstractmethod, ABC

import numpy as np

from src.domain.domain import ProblemType


class Metric:

    def __init__(self, problem_type: str):
        self.problem_type = problem_type

    @abstractmethod
    def calculate(self, y_true: np.array, y_hat: np.array) -> float:
        ...

    def applicable_to(self, problem_type: str):
        return self.problem_type == problem_type


class RMSE(Metric):

    def __init__(self):
        super().__init__(ProblemType.REGRESSION)

    def calculate(self, y_true: np.array, y_hat: np.array) -> float:
        return np.sqrt(MSE().calculate(y_true, y_hat))


class MSE(Metric):

    def __init__(self):
        super().__init__(ProblemType.REGRESSION)

    def calculate(self, y_true: np.array, y_hat: np.array) -> float:
        return np.square(y_hat - y_true).mean()


class Accuracy(Metric):

    def __init__(self):
        super().__init__(ProblemType.CLASSIFICATION)

    def calculate(self, y_true: np.array, y_hat: np.array) -> float:
        return np.equal(y_true, y_hat).mean()
