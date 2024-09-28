import numpy as np
from numpy.core.multiarray import array as array
from math import cos, sin, pi
from .base import GrayModel


class EFGVM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int):
        super().gvm_fit(data, window_size)

        # calculating errors based on true data points
        self._residuals(window_size)

        N = len(self.residuals)
        predicted = self.predicted.copy()
        self.predicted.clear()

        # n = len(self.windowed_data) - 1  # Adjusting index for residuals
        T = N  # Total number of points
        z = int(N / 2) - 1  # Number of Fourier terms

        P = []  # np.empty((N, N))
        for i in range(2, N + 2):
            row = [0.5]
            for j in range(1, z + 2):
                row.append(cos(i * 2 * pi * j / T))
                row.append(sin(i * 2 * pi * j / T))
            P.append(row)

        P = np.array(P)
        self.residuals = np.array(self.residuals)
        C = np.linalg.inv(P.T @ P) @ P.T @ self.residuals
        # C = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(P),P)), np.transpose(P)), np.transpose([self.residuals]))

        a_0 = C[0]
        A_i = []
        B_i = []
        for i in range(1, len(C)):
            if i % 2 == 1:
                A_i.append(C[i])
            else:
                B_i.append(C[i])

        corrected_predictions = []
        for k in range(1, N + 1):
            correction = (1 / 2) * a_0
            for i in range(z):
                correction += A_i[i] * np.cos(2 * np.pi * k / T * (i + 1))
                correction += B_i[i] * np.sin(2 * np.pi * k / T * (i + 1))
            corrected_prediction = predicted[k] - correction
            corrected_predictions.append(corrected_prediction)

        self.predicted = corrected_predictions.copy()

        #  for corr in self.corrections:

        # calculating RPE and ARPE
        self._score(window_size)