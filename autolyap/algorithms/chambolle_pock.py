import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class ChambollePock(Algorithm):
    def __init__(self, tau, sigma, theta):
        super().__init__(2, 2, [1, 1], [1, 2], [])
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
    
    def set_tau(self, tau):
        self.tau = tau
    
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_theta(self, theta):
        self.theta = theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1, -self.tau],
                      [0, 0]])
        B = np.array([[-self.tau, 0],
                      [0, 1]])
        C = np.array([[1, -self.tau], 
                      [1, 1/self.sigma - self.tau*(1+self.theta)]])
        D = np.array([[-self.tau, 0], 
                      [-self.tau*(1+self.theta), -1/self.sigma]])
        return (A, B, C, D)
