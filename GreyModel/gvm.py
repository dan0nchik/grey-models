import numpy as np
from numpy.core.multiarray import array as array
from .base import GrayModel


class GVM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        # split data into windows
        windowed_data = self._preprocess(data, window_size)

        # iterate through windows

        for X0 in windowed_data:
            n = len(X0)
            X1 = X0.cumsum()  # Applying AGO
            Z1 = np.empty(n - 1)  # generated mean sequence of X1
            for i in range(1, n):
                Z1[i - 1] = 0.5 * X1[i - 1] + 0.5 * X1[i]

            B = np.empty((n - 1, 2))  # matrix for parameters a,b

            for i in range(n - 1):
                B[i][0] = -Z1[i]
                B[i][1] = Z1[i] ** 2

            Y = X0[1:].T
            AB = np.linalg.inv(B.T @ B) @ B.T @ Y  # OLS
            a = AB[0]
            b = AB[1]

            # predict next step
            x = (a * X0[0]) / (b * X0[0] + (a - b * X0[0]) * np.exp(a * (n)))
            self.predicted.append(x)

        # calculating errors based on true data points
        data_true = data[window_size : len(self.predicted) + window_size]
        for i in range(len(data_true)):
            self.residuals.append(data_true[i] - self.predicted[i])

        # calculating RPE and ARPE
        rpe = []
        for i in range(len(data_true)):
            rpe.append(abs(self.residuals[i]) / data_true[i] * 100)
        self.arpe = (1 / (len(data_true) - 1)) * sum(rpe) / 100
