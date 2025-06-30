import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class ProximalPoint(Algorithm):
    def __init__(self, gamma):
        super().__init__(1, 1, [1], [1], [])
        self.gamma = gamma
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[-self.gamma]])
        C = np.array([[1]])
        D = np.array([[-self.gamma]])
        return (A, B, C, D)