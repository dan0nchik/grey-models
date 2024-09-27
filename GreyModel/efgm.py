import numpy as np
from numpy.core.multiarray import array as array
from math import cos, sin, pi
from .base import GrayModel


class EFGM(GrayModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: np.array, window_size: int, predict_step: int = 1):
        super().fit(data, window_size)
        for X0 in self.windowed_data:
            n = window_size
            Z1 = self._get_Z1(X0, window_size)

            B = np.empty((n - 1, 2))  # matrix for parameters a,b

            for i in range(n - 1):
                B[i][0] = -Z1[i]
                B[i][1] = 1

            Y = X0[1:].T
            AB = np.linalg.inv(B.T @ B) @ B.T @ Y  # OLS
            a = AB[0]
            b = AB[1]
            
            # predict next step
            # x = (a * X0[0]) / (b * X0[0] + (a - b * X0[0]) * np.exp(a * (n)))
            # 
            x = (X0[0] - (b/a))*np.exp(-a*(n-1))*(1-np.exp(a))
            # x = (a*X0[0]*(a-b*X0[0])*(1-np.exp(a))*np.exp(a*(n-2)))/((b*X0[0]+(a-b*X0[0]*np.exp(a*(n-1))))*(b*X0[0]+(a-b*X0[0]*np.exp(a*(n-2)))))
            self.predicted.append(x)

        # calculating errors based on true data points
        self._residuals(window_size)
        
        N = len(self.residuals)
        
        prediction_gvm = self.predicted.copy()
        self.predicted.clear()
        
        for window in range(0, len(self.windowed_data)):
            initially_predicted = prediction_gvm[window]
            X0 = self.windowed_data[window]
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
            error = 0.5*a_0 + summa
            self.predicted.append(initially_predicted - error)
        
        self.residuals.clear()
        self._residuals(window_size)

        # calculating RPE and ARPE
        self._score(window_size)