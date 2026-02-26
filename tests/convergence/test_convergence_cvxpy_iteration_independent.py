import numpy as np
import pytest

from autolyap import IterationIndependent
from autolyap.algorithms import GradientMethod, ProximalPoint
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex, StronglyConvex


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def _rho_tolerance_for_solver(cvxpy_convergence_solver_options) -> float:
    if cvxpy_convergence_solver_options.cvxpy_solver == "SCS":
        return 1e-2
    return 5e-4


def test_convergence_gradient_method_rho_matches_theory_cvxpy(
    cvxpy_convergence_solver_options,
):
    mu = 1.0
    L = 4.0
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=1.0)
    tol = _rho_tolerance_for_solver(cvxpy_convergence_solver_options)

    gammas = np.linspace(0.1, 0.4, 6)
    for gamma in gammas:
        algorithm.set_gamma(float(gamma))
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(algorithm)
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
            solver_options=cvxpy_convergence_solver_options,
        )
        assert result["status"] == "feasible"
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = max(gamma * L - 1.0, 1.0 - gamma * mu) ** 2
        assert abs(rho_al - rho_theoretical) < tol


def test_convergence_proximal_point_rho_matches_theory_cvxpy(
    cvxpy_convergence_solver_options,
):
    mu = 1.0
    problem = InclusionProblem([StronglyConvex(mu)])
    algorithm = ProximalPoint(gamma=1.0)
    tol = _rho_tolerance_for_solver(cvxpy_convergence_solver_options)

    gammas = np.linspace(0.1, 0.9, 6)
    for gamma in gammas:
        algorithm.set_gamma(float(gamma))
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(algorithm)
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
            solver_options=cvxpy_convergence_solver_options,
        )
        assert result["status"] == "feasible"
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = (1.0 / (1.0 + gamma * mu)) ** 2
        assert abs(rho_al - rho_theoretical) < tol
