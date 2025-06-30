import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class TripleMomentum(Algorithm):
    # https://ieeexplore.ieee.org/document/7967721
    # Triple momentum method
    def __init__(self, sigma, beta):
        super().__init__(2, 1, [1], [1], [])
        self.sigma = sigma
        self.beta = beta
    
    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_beta(self, beta):
        self.beta = beta

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.sigma/self.beta
        
        alpha = (2-np.sqrt(q))/self.beta
        _beta = (1-np.sqrt(q))**2/(1+np.sqrt(q))
        gamma = (1-np.sqrt(q))**2/((2-np.sqrt(q))*(1+np.sqrt(q)))    
        
        A = np.array([[1+_beta,-_beta],[1,0]])
        B = np.array([[-alpha],[0]])
        C = np.array([[1+gamma,-gamma]])
        D = np.array([[0]])
        
        return (A, B, C, D)