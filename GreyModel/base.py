import numpy as np


class GrayModel:
    def __init__(self) -> None:
        self.data = []
        self.residuals = []
        self.predicted = []
        self.arpe = None

    def _preprocess(self, data, window_size):
        return np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)

    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        pass

    def predict(self, X: np.array):
        pass

    def score(self, y_true: np.array, y_pred: np.array):
        pass

    def get_arpe(self):
        return self.arpe

    def get_residuals(self):
        return self.residuals
