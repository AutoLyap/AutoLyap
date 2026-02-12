import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    giselsson_boyd_delta,
    giselsson_boyd_rate_sq,
)


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def test_convergence_douglas_rachford_function_smooth_strong_plus_convex_giselsson_boyd_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    mu = 1.0
    L = 2.0
    lambda_value = 1.0
    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L), Convex()])
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="function")

    for gamma in np.linspace(0.1, 3.0, 3):
        delta = giselsson_boyd_delta(L, mu, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = bisection_rho(
            problem,
            algorithm,
            solver_options=cvxpy_clarabel_solver_options,
        )
        assert result["success"]
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = giselsson_boyd_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=2e-3)
