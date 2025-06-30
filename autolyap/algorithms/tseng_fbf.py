import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class TsengFBF(Algorithm):
    def __init__(self, gamma, theta):
        super().__init__(1, 2, [2, 1], [], [1, 2]) 
        self.gamma = gamma
        self.theta = theta
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_theta(self, theta):
        self.theta = theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[0, -self.gamma*self.theta, -self.gamma*self.theta]])
        C = np.array([[1],
                      [1],
                      [1]])
        D = np.array([[0, 0, 0], 
                      [-self.gamma, 0, -self.gamma],
                      [-self.gamma, 0, -self.gamma]])
        return (A, B, C, D)
