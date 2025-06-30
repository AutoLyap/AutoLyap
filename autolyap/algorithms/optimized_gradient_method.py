import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm

class OptimizedGradientMethod(Algorithm):
    def __init__(self, L, K):
        super().__init__(2, 1, [1], [1], [])
        self.L = L
        self.K = K
    
    def set_L(self, L):
        self.L = L
    
    def set_K(self, K):
        self.K = K
    
    def _compute_theta(self, k: int, K: int) -> float:
        if k < 0 or k > K:
            raise ValueError("k must be a non-negative integer and less than or equal to K.")
        
        theta = 1.0 
        for i in range(1, k + 1):
            if i == K:
                theta = (1 + math.sqrt(1 + 8 * theta ** 2)) / 2
            else:
                theta = (1 + math.sqrt(1 + 4 * theta ** 2)) / 2
        return theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if k < self.K:
            theta_k = self._compute_theta(k, self.K)
            theta_kp1 = self._compute_theta(k + 1, self.K)
            
            A = np.array([[1+(theta_k-1)/theta_kp1, (1-theta_k)/theta_kp1],
                          [1, 0]])
            
            B = np.array([[-(1+(2*theta_k-1)/theta_kp1)/self.L],
                          [-1/self.L]])
            
            C = np.array([[1, 0]])
            
            D = np.array([[0]])
        
        elif k == self.K:
            A = np.array([[0, 0],
                          [0, 0]])
            
            B = np.array([[0],
                          [0]])
            
            C = np.array([[1, 0]])
            
            D = np.array([[0]])
        
        else:
            raise ValueError("k must be less than or equal to K.")
        
        return (A, B, C, D)
