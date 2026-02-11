import numpy as np
import pytest

from autolyap import IterationIndependent
from autolyap.algorithms import Algorithm
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def _docs_tolerance(solver_name: str) -> float:
    return 1e-3 if solver_name == "CLARABEL" else 2e-2

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


def _validate_parameters(mu: float, L: float, gamma: float) -> None:
    if not (0.0 < mu < L):
        raise ValueError(f"Invalid parameters: require 0 < mu < L. Got mu={mu}, L={L}.")

    gamma_max = 2.0 / L
    if not (0.0 < gamma <= gamma_max):
        raise ValueError(
            f"Invalid parameters: require 0 < gamma <= 2/L. Got gamma={gamma}, 2/L={gamma_max}."
        )


def test_docs_example_proximal_gradient_workflow_matches_theory_cvxpy(
    cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
):
    mu = 1.0
    L = 4.0
    gamma = 2.0 / (L + mu)
    _validate_parameters(mu=mu, L=L, gamma=gamma)

    problem = InclusionProblem([
        SmoothStronglyConvex(mu, L),
        Convex(),
    ])
    algorithm = ProximalGradientMethod(gamma=gamma)

    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm,
        i=1,
        j=1,
    )
    result = IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        p=p,
        t=t,
        S_equals_T=True,
        s_equals_t=True,
        remove_C3=True,
        solver_options=cvxpy_open_source_solver_options,
    )

    assert result["success"]
    rho_autolyap = result["rho"]
    assert rho_autolyap is not None
    rho_taylor = max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2
    assert abs(rho_autolyap - rho_taylor) < _docs_tolerance(cvxpy_open_source_solver_name)


@pytest.mark.parametrize(
    "mu,L,gamma",
    [
        (1.0, 1.0, 0.1),   # mu !< L
        (2.0, 1.0, 0.1),   # mu > L
        (1.0, 4.0, 0.0),   # gamma <= 0
        (1.0, 4.0, 0.6),   # gamma > 2/L
    ],
)
def test_docs_example_proximal_gradient_parameter_validation_raises(mu, L, gamma):
    with pytest.raises(ValueError):
        _validate_parameters(mu=mu, L=L, gamma=gamma)
