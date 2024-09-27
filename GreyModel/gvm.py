from matplotlib import pyplot as plt
import numpy as np
from numpy.core.multiarray import array as array
from .base import GrayModel


class GVM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int):

        super().gvm_fit(data, window_size)

        # calculating errors based on true data points
        self._residuals(window_size)

        # calculating RPE and ARPE
        self._score(window_size)
