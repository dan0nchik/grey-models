import numpy as np
from numpy.core.multiarray import array as array
from .base import GrayModel
import math

class GM(GrayModel):
    def __init__(self) -> None:
        super().__init__()
        self.predicted_fourier = []
        self.farpe = 0


    def get_farpe(self):
        return self.farpe


    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        windowed_data = self._preprocess(data, window_size)

        for X0 in windowed_data:
            X1 = X0.cumsum()  # Applying AGO
            Z1 = np.empty(window_size - 1)  # generated mean sequence of X1
            for i in range(1, window_size):
                Z1[i - 1] = 0.5 * X1[i - 1] + 0.5 * X1[i]

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
            # x1 = (X0[1] - (b / a)) * (np.e ** ((-a) * window_size)) + (b / a)
            self.predicted.append(x)

        # calculating errors based on true data points
        data_true = data[window_size:]
        E = np.empty(len(data) - window_size)
        for i in range(len(data_true)):
            self.residuals.append(data_true[i] - self.predicted[i])
            E[i] = data_true[i] - self.predicted[i]


        # calculating RPE and ARPE
        rpe = []
        for i in range(len(data_true)):
            rpe.append(abs(self.residuals[i]) / data_true[i] * 100)
        self.arpe = (1 / (len(data_true) - 1)) * sum(rpe) / 100

        # n = len(data) - window_size
        # T = n - 1
        # z = (n - 1 // 2) - 1
        # P = np.empty((T, 2 * z + 1))
        # for i in range(T):
        #     P[i][0] = 1 / 2
        #     for j in range(2 * z):
        #         if j % 2 == 0:
        #             P[i][j + 1] = math.cos((i + 2) * 2 * math.pi * (j // 2 + 1) / T)
        #         else:
        #             P[i][j + 1] = math.sin((i + 2) * 2 * math.pi * (j // 2 + 1) / T)
        #
        #
        # C = np.linalg.inv(P.T @ P) @ P.T @ E
        # E1 = np.empty(len(self.data) - window_size)
        # for i in range(len(self.data) - window_size):
        #     E1[i] = (1 / 2) * C[0]
        #     for j in range(2 * z):
        #         if j % 2 == 0:
        #             E1[i] += C[j] * math.cos(2 * math.pi * (j // 2 + 1) * i / T)
        #         else:
        #             E1[i] += C[j] * math.sin(2 * math.pi * (j // 2 + 1) * i / T)
        # for i in range(len(self.predicted)):
        #     self.predicted_fourier.append(self.predicted[i] - E1[i])
        #
        # E_f = np.empty(len(self.data) - window_size)
        # for i in range(len(self.predicted_fourier)):
        #     E_f[i] = data_true[i] - self.predicted_fourier[i]
        #
        # frpe = []
        # for i in range(len(data_true)):
        #     frpe.append(abs(E_f[i]) / data_true[i] * 100)
        # self.farpe = (1 / (len(data_true) - 1)) * sum(frpe) / 100









