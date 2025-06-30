import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm

class ITEM(Algorithm):
    def __init__(self, mu, L):
        super().__init__(2, 1, [1], [1], [])
        self.mu = mu
        self.L = L
    
    def set_L(self, L):
        self.L = L

    def set_mu(self, mu):
        self.mu = mu
    
    def get_A(self, k: int) -> float:
        q = self.mu / self.L
        A = 0.0 
        for _ in range(k):
            A = ((1 + q) * A + 2 * (1 + math.sqrt((1 + A) * (1 + q * A)))) / ((1 - q) ** 2)
        return A

    def compute_beta(self, k: int) -> float:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        return A_k / ((1 - q) * A_k1)

    def compute_delta(self, k: int) -> float:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        numerator = ((1 - q) ** 2) * A_k1 - (1 + q) * A_k
        denominator = 2 * (1 + q + q * A_k)
        return numerator / denominator

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        beta = self.compute_beta(k)
        delta = self.compute_delta(k)

        A = np.array([[beta, 1-beta],
                      [q*beta*delta, 1-q*beta*delta]])
        
        B = np.array([[-1/self.L],
                      [-delta/self.L]])
        
        C = np.array([[beta, 1-beta]])
        
        D = np.array([[0]])
        
        return (A, B, C, D)