import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class HeavyBallMethod(Algorithm):
    def __init__(self, gamma, delta):
        super().__init__(2, 1, [1], [1], [])
        self.gamma = gamma
        self.delta = delta
    
    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_delta(self, delta):
        self.delta = delta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1+self.delta, -self.delta], [1, 0]]) 
        B = np.array([[-self.gamma],[0]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return (A, B, C, D)