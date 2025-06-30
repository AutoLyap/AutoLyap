import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class DavisYin(Algorithm):
    # https://arxiv.org/pdf/1504.01032.pdf
    def __init__(self, gamma, lambda_value):
        super().__init__(1, 3, [1, 1, 1], [1, 2, 3], [])
        self.gamma = gamma
        self.lambda_value = lambda_value
    
    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = -self.gamma*self.lambda_value*np.array([[1,1,1]])
        C = np.array([[1],[1],[1]])
        D = -self.gamma*np.array([[1,0,0],[1,0,0],[2,1,1]])
        return (A, B, C, D)