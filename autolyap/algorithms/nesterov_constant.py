import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class NesterovConstant(Algorithm):
    # https://link.springer.com/book/10.1007/978-3-319-91578-4 
    # See Chapter 2.2: Constant step scheme, III
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
        
        A = np.array([[2/(1+np.sqrt(q)),-(1-np.sqrt(q))/(1+np.sqrt(q))],[1,0]])
        B = np.array([[-1/beta],[0]])
        C = np.array([[2/(1+np.sqrt(q)),-(1-np.sqrt(q))/(1+np.sqrt(q))]])
        D = np.array([[0]])
        return (A, B, C, D)