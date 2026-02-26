import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    dr_smooth_strongly_convex_plus_convex_delta,
    dr_smooth_strongly_convex_plus_convex_rate_sq,
)


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def _rho_tolerance_for_solver(cvxpy_convergence_solver_options) -> float:
    if cvxpy_convergence_solver_options.cvxpy_solver == "SCS":
        return 1e-2
    return 2e-3


def test_convergence_douglas_rachford_function_smooth_strongly_convex_plus_convex_cvxpy(
    cvxpy_convergence_solver_options,
):
    mu = 1.0
    L = 2.0
    lambda_value = 1.0
    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L), Convex()])
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="function")

    for gamma in np.linspace(0.1, 3.0, 3):
        delta = dr_smooth_strongly_convex_plus_convex_delta(L, mu, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = bisection_rho(
            problem,
            algorithm,
            solver_options=cvxpy_convergence_solver_options,
        )
        assert result["status"] == "feasible"
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = dr_smooth_strongly_convex_plus_convex_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(
            rho_theoretical,
            abs=_rho_tolerance_for_solver(cvxpy_convergence_solver_options),
        )
