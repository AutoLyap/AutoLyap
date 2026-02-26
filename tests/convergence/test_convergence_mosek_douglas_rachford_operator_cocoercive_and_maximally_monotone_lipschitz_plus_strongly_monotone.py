import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    InclusionProblem,
    StronglyMonotone,
    LipschitzOperator,
    Cocoercive,
    MaximallyMonotone,
)
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    dr_cocoercive_plus_strongly_monotone_rate,
    dr_maximally_monotone_lipschitz_plus_strongly_monotone_rate,
)
from tests.shared.mosek_utils import require_mosek_license


pytestmark = pytest.mark.mosek


@pytest.fixture(scope="module", autouse=True)
def _require_mosek():
    require_mosek_license()


def test_convergence_douglas_rachford_operator_cocoercive_plus_strongly_monotone_lambda_sweep(
    mosek_convergence_solver_options,
):
    mu = 1.0
    beta = 1.0
    gamma = 1.0
    problem = InclusionProblem([Cocoercive(beta=beta), StronglyMonotone(mu=mu)])
    algorithm = DouglasRachford(gamma=gamma, lambda_value=2.0, type="operator")

    for lambda_value in np.linspace(0.01, 1.99, 5):
        algorithm.set_lambda(float(lambda_value))
        result = bisection_rho(
            problem,
            algorithm,
            solver_options=mosek_convergence_solver_options,
        )
        assert result["status"] == "feasible"
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = dr_cocoercive_plus_strongly_monotone_rate(mu, beta, lambda_value, gamma) ** 2
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)


def test_convergence_douglas_rachford_operator_maximally_monotone_lipschitz_plus_strongly_monotone_lambda_sweep(
    mosek_convergence_solver_options,
):
    mu = 1.0
    L = 1.0
    gamma = 1.0
    problem = InclusionProblem(
        [[MaximallyMonotone(), LipschitzOperator(L=L)], StronglyMonotone(mu=mu)]
    )
    algorithm = DouglasRachford(gamma=gamma, lambda_value=2.0, type="operator")

    for lambda_value in np.linspace(0.01, 1.99, 5):
        algorithm.set_lambda(float(lambda_value))
        result = bisection_rho(
            problem,
            algorithm,
            solver_options=mosek_convergence_solver_options,
        )
        assert result["status"] == "feasible"
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = dr_maximally_monotone_lipschitz_plus_strongly_monotone_rate(mu, lambda_value, L, gamma) ** 2
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)
