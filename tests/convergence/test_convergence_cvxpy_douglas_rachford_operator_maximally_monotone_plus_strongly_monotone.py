import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    Cocoercive,
    InclusionProblem,
    LipschitzOperator,
    MaximallyMonotone,
    StronglyMonotone,
)
from tests.convergence.convergence_douglas_rachford_utils import (
    bisection_rho,
    dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta,
    dr_maximally_monotone_plus_strongly_monotone_lipschitz_rate_sq,
    dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta,
    dr_maximally_monotone_plus_strongly_monotone_cocoercive_rate_sq,
)


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def test_convergence_douglas_rachford_operator_maximally_monotone_plus_strongly_monotone_lipschitz_cvxpy(
    cvxpy_convergence_solver_options,
):
    mu = 1.0
    L = 2.0
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), LipschitzOperator(L=L)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.1, 3.0, 3):
        delta = dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta(mu, L, gamma)
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
        rho_theoretical = dr_maximally_monotone_plus_strongly_monotone_lipschitz_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=2e-3)


def test_convergence_douglas_rachford_operator_maximally_monotone_plus_strongly_monotone_cocoercive_cvxpy(
    cvxpy_convergence_solver_options,
):
    mu = 1.0
    beta = 0.5
    L = 1.0 / beta
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), Cocoercive(beta=beta)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.1, 3.0, 3):
        delta = dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta(mu, L, gamma)
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
        rho_theoretical = dr_maximally_monotone_plus_strongly_monotone_cocoercive_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=2e-3)
