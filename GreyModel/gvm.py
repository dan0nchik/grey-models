from matplotlib import pyplot as plt
import numpy as np
from numpy.core.multiarray import array as array
from .base import GrayModel


class GVM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int):
        # split data into windows
        super().fit(data, window_size)
        # iterate through windows
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
            # x = (X0[1] - (b / a)) * (np.e ** ((-a) * window_size)) * (1 - np.e ** a)
            self.predicted.append(x)
        # calculating errors based on true data points
        self._residuals(window_size)

        # calculating RPE and ARPE
        self._score(window_size)
