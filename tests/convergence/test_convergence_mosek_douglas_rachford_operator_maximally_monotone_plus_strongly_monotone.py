import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    InclusionProblem,
    MaximallyMonotone,
    StronglyMonotone,
    LipschitzOperator,
    Cocoercive,
)
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta,
    dr_maximally_monotone_plus_strongly_monotone_lipschitz_rate_sq,
    dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta,
    dr_maximally_monotone_plus_strongly_monotone_cocoercive_rate_sq,
)
from tests.shared.mosek_utils import require_mosek_license


pytestmark = pytest.mark.mosek


@pytest.fixture(scope="module", autouse=True)
def _require_mosek():
    require_mosek_license()


def test_convergence_douglas_rachford_operator_maximally_monotone_plus_strongly_monotone_lipschitz():
    mu = 1.0
    L = 2.0
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), LipschitzOperator(L=L)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta(mu, L, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = bisection_rho(problem, algorithm)
        assert result["status"] == "feasible"
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = dr_maximally_monotone_plus_strongly_monotone_lipschitz_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-5)


def test_convergence_douglas_rachford_operator_maximally_monotone_plus_strongly_monotone_cocoercive():
    mu = 1.0
    beta = 0.5
    L = 1.0 / beta
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), Cocoercive(beta=beta)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta(mu, L, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = bisection_rho(problem, algorithm)
        assert result["status"] == "feasible"
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = dr_maximally_monotone_plus_strongly_monotone_cocoercive_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)
