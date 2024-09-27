from matplotlib import pyplot as plt
import numpy as np
from numpy.core.multiarray import array as array
from .base import GrayModel


class GM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        super().fit(data, window_size)
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

        # calculating errors based on true data points
        self._residuals(window_size)

        self._score(window_size)