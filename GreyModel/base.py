import numpy as np


class GrayModel:
    def __init__(self) -> None:
        self.data = []
        self.windowed_data = []
        self.residuals = []
        self.predicted = []
        self.arpe = None

    def _preprocess(self, data, window_size):
        return np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)

    def _get_Z1(self, X0, window_size):
        X1 = X0.cumsum()  # Applying AGO
        Z1 = np.empty(window_size - 1)  # generated mean sequence of X1
        for i in range(1, window_size):
            Z1[i - 1] = 0.5 * X1[i - 1] + 0.5 * X1[i]
        return Z1

    def _residuals(self, window_size):
        data_true = self.data[window_size:]
        for i in range(len(data_true)):
            self.residuals.append(data_true[i] - self.predicted[i])

    def _score(self, window_size):
        data_true = self.data[window_size:]
        rpe = []
        for i in range(len(data_true)):
            rpe.append(abs(self.residuals[i]) / data_true[i] * 100)
        self.arpe = (1 / (len(data_true) - 1)) * sum(rpe)

    def fit(self, data: np.array, window_size: int):
        self.data = data
        self.windowed_data = self._preprocess(data, window_size)

    def predict(self, X: np.array):
        pass

    def get_arpe(self):
        return self.arpe

    def get_residuals(self):
        return self.residuals
