import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex, Convex
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    giselsson_boyd_delta,
    giselsson_boyd_rate_sq,
)
from tests.shared.mosek_utils import require_mosek_license


pytestmark = pytest.mark.mosek


@pytest.fixture(scope="module", autouse=True)
def _require_mosek():
    require_mosek_license()


def test_convergence_douglas_rachford_function_smooth_strong_plus_convex_giselsson_boyd():
    mu = 1.0
    L = 2.0
    lambda_value = 1.0
    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L), Convex()])
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="function")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = giselsson_boyd_delta(L, mu, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = giselsson_boyd_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)
