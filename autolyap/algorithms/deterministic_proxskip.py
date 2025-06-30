import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class ProxSkip(Algorithm):
    # https://arxiv.org/pdf/2202.09357
    def __init__(self, gamma, L):
        self.set_gamma(gamma)  
        self.set_L(L) 
        super().__init__(2, 2, [self.L, 1], [1, 2], [])

    def set_gamma(self, gamma):
        if not isinstance(gamma, (int, float)) or gamma <= 0:
            raise ValueError("gamma must be a real number greater than 0")
        self.gamma = gamma

    def set_L(self, L):
        if not isinstance(L, int) or L < 1:
            raise ValueError("L must be an integer greater than or equal to 1")
        self.L = L
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1, self.gamma*(1-self.L)],
                      [0, 0]])
        B = np.vstack((np.hstack((-self.gamma * np.ones((1,self.L)), np.array([[  -self.gamma * (2*self.L - 1) ]]))),
                       np.hstack((np.zeros((1, self.L)), np.array([[-1]])))))
        C = np.vstack((np.array([[1, 0]]),
                       np.hstack((np.ones((self.L, 1)), self.gamma * (1 - self.L) * np.ones((self.L, 1))))))
        D = self.create_D()
        return (A, B, C, D)

    def create_D(self):
        D = np.zeros((self.L+1, self.L+1))
        for i in range(1, self.L):
            D[i, :i] = -self.gamma  
            D[i, -1] = -self.gamma*( i - 1 + self.L)
        D[-1, 0] = -self.gamma  
        D[-1, -1] = -self.gamma * self.L
        return D