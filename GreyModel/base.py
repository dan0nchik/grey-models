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

    def gm_fit(self, data: np.array, window_size: int):
        self.data = data
        self.windowed_data = self._preprocess(data, window_size)
        for X0 in self.windowed_data:

            Z1 = self._get_Z1(X0, window_size)

            B = np.empty((window_size - 1, 2))  # matrix for parameters a,b

            for i in range(window_size - 1):
                B[i][0] = -Z1[i]
                B[i][1] = 1

            Y = X0[1:].T
            AB = np.linalg.inv(B.T @ B) @ B.T @ Y  # OLS
            a = AB[0]
            b = AB[1]

            # predict next step
            x = (X0[1] - (b / a)) * (np.e ** ((-a) * window_size)) * (1 - np.e ** a)
            self.predicted.append(x)

    def gvm_fit(self, data: np.array, window_size: int):
        self.data = data
        self.windowed_data = self._preprocess(data, window_size)
        for X0 in self.windowed_data:
            Z1 = self._get_Z1(X0, window_size)
            B = np.empty((window_size - 1, 2))  # matrix for parameters a,b

            for i in range(window_size - 1):
                B[i][0] = -Z1[i]
                B[i][1] = Z1[i] ** 2

            # exit()
            Y = X0[1:].T
            AB = np.linalg.inv(B.T @ B) @ B.T @ Y  # OLS
            a = AB[0]
            b = AB[1]

            nomenator = (
                    a
                    * X0[0]
                    * (a - b * X0[0])
                    * (np.exp(a * (window_size - 2)) - np.exp(a * (window_size - 1)))
            )

            denominator = (b * X0[0] + (a - b * X0[0]) * np.exp(a * (window_size - 1))) * (
                    b * X0[0] + (a - b * X0[0]) * np.exp(a * (window_size - 2))
            )

            x = nomenator / denominator
            self.predicted.append(x)

    def get_arpe(self):
        return self.arpe

    def get_residuals(self):
        return self.residuals
