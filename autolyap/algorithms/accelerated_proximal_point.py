import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class AcceleratedProximalPoint(Algorithm):
    def __init__(self, gamma, type: str = "operator"):
        if type == "operator":
            super().__init__(3, 1, [1], [], [1])
        elif type == "function":
            super().__init__(3, 1, [1], [1], [])
        else:
            raise ValueError("type must be either 'operator' or 'function'")
        self.gamma = gamma
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lambda_var = k / (k + 2)
        A = np.array([[0, 1 , 0], 
                      [-2*lambda_var, 1+lambda_var, lambda_var],
                      [0, 1, 0]])
        B = np.array([[-self.gamma],
                      [-self.gamma*(1+lambda_var)],
                      [0]])
        C = np.array([[0, 1, 0]])
        D = np.array([[-self.gamma]])
        return (A, B, C, D)