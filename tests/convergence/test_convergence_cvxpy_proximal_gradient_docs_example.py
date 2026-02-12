import pytest

from autolyap import IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex
from tests.convergence.convergence_proximal_gradient_docs_utils import (
    ProximalGradientMethod,
    validate_parameters,
)


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")

def test_docs_example_proximal_gradient_workflow_matches_theory_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    mu = 1.0
    L = 4.0
    gamma = 2.0 / (L + mu)
    validate_parameters(mu=mu, L=L, gamma=gamma)

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
        solver_options=cvxpy_clarabel_solver_options,
    )

    assert result["success"]
    rho_autolyap = result["rho"]
    assert rho_autolyap is not None
    rho_taylor = max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2
    assert abs(rho_autolyap - rho_taylor) < 1e-3


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
        validate_parameters(mu=mu, L=L, gamma=gamma)
