import numpy as np

from autolyap.algorithms import Algorithm


class ProximalGradientMethod(Algorithm):
    """Test-local copy of the algorithm defined in docs/source/examples/proximal_gradient.md."""

    def __init__(self, gamma: float):
        super().__init__(n=1, m=2, m_bar_is=[1, 1], I_func=[1, 2], I_op=[])
        self.set_gamma(gamma)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def get_ABCD(self, k: int):
        A = np.array([[1.0]])
        B = np.array([[-self.gamma, -self.gamma]])
        C = np.array([[1.0], [1.0]])
        D = np.array([
            [0.0, 0.0],
            [-self.gamma, -self.gamma],
        ])
        return A, B, C, D


def validate_parameters(mu: float, L: float, gamma: float) -> None:
    if not (0.0 < mu < L):
        raise ValueError(f"Invalid parameters: require 0 < mu < L. Got mu={mu}, L={L}.")

    gamma_max = 2.0 / L
    if not (0.0 < gamma <= gamma_max):
        raise ValueError(
            f"Invalid parameters: require 0 < gamma <= 2/L. Got gamma={gamma}, 2/L={gamma_max}."
        )
