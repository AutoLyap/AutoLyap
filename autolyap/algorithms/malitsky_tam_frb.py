import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class MalitskyTamFRB(Algorithm):
    def __init__(self, gamma):
        super().__init__(2, 2, [2, 1], [], [1, 2])
        self.gamma = gamma
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1, 0],
                      [1, 0]])
        B = np.array([[-2*self.gamma, self.gamma, -self.gamma],
                      [0, 0, 0]])
        C = np.array([[1, 0],
                      [0, 1],
                      [1, 0]])
        D = np.array([[0, 0, 0], 
                      [0, 0, 0],
                      [-2*self.gamma, self.gamma, -self.gamma]])
        return (A, B, C, D)