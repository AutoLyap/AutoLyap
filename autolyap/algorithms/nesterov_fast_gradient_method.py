import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class NesterovFastGradientMethod(Algorithm):
    def __init__(self, gamma):
        super().__init__(2, 1, [2], [1], [])
        self.gamma = gamma
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lambda_var = 1
        for _ in range(0, k + 1):
            lambda_var_prev = lambda_var
            lambda_var = (1 + np.sqrt(1 + 4 * lambda_var ** 2)) / 2

        alpha = (lambda_var_prev - 1) / lambda_var

        A = np.array([[1+alpha, -alpha],
                      [1, 0]])
        
        B = np.array([[-self.gamma, 0],
                      [0, 0]])
        
        C = np.array([[1+alpha, -alpha],
                      [1, 0]])
        
        D = np.array([[0, 0],
                      [0, 0]])
        
        return (A, B, C, D)
