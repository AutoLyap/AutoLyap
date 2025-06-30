import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class DouglasRachford(Algorithm):
    def __init__(self, gamma, lambda_value, operator_version = True):
        if operator_version:
            super().__init__(1, 2, [1, 1], [], [1, 2])
        else:
            super().__init__(1, 2, [1, 1], [1, 2], [])
        self.gamma = gamma
        self.lambda_value = lambda_value
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[-self.gamma*self.lambda_value, -self.gamma*self.lambda_value]])
        C = np.array([[1], 
                      [1]])
        D = np.array([[-self.gamma, 0], 
                      [-2*self.gamma, -self.gamma]])
        return (A, B, C, D)