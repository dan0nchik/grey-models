import numpy as np
from numpy.core.multiarray import array as array
from math import cos, sin, pi
from .base import GrayModel


class TFGVM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        # split data into windows
        super().gvm_fit(data, window_size)

        # calculating errors based on true data points
        self._residuals(window_size)
        
        N = len(self.residuals)
        self.predicted.clear()
        
        for X0 in self.windowed_data:
            n = len(X0) - 1
            z = int((N-1)/2) - 1
            T = N - 1
                
            P = []
            for i in range (2, N+2):
                row = [0.5]
                for j in range(1, z+2):
                    row.append(cos(i*2*pi*j/T))
                    row.append(sin(i*2*pi*j/T))
                P.append(row)
                
            C = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(P),P)), np.transpose(P)), np.transpose([self.residuals]))
            
            a_0 = C[0]
            A_i = []
            B_i = []
            for i in range(1, len(C)):
                if i%2 == 1:
                    A_i.append(C[i])
                else:
                    B_i.append(C[i])
            summa = 0
            for i in range(0, z):
                summa += A_i[i]*cos(n*2*pi*(i+1)/T) + B_i[i]*sin(n*2*pi*(i+1)/T)
            q = 0.5*a_0 + summa
            self.predicted.append((n-q))
        
        self.residuals.clear()
        self._residuals(window_size)

        # calculating RPE and ARPE
        self._score(window_size)