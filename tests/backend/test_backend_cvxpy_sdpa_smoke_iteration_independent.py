# Smoke checks for CVXPY SDPA backends (regular and multiprecision profiles).

import numpy as np

from autolyap import IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem


def test_iteration_independent_verify_with_cvxpy_sdpa_backend_smoke(
    tiny_functional_algorithm, cvxpy_sdpa_solver_options
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        rho=1.0,
        solver_options=cvxpy_sdpa_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["status"] == "feasible":
        certificate = result["certificate"]
        assert certificate is not None
        assert "Q" in certificate
        assert "S" in certificate
        assert "multipliers" in certificate


def test_iteration_independent_verify_with_cvxpy_sdpa_multiprecision_backend_smoke(
    tiny_functional_algorithm, cvxpy_sdpa_multiprecision_solver_options
):
    assert cvxpy_sdpa_multiprecision_solver_options.cvxpy_solver_params is not None
    assert cvxpy_sdpa_multiprecision_solver_options.cvxpy_solver_params["mpfPrecision"] == 512

    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        rho=1.0,
        solver_options=cvxpy_sdpa_multiprecision_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["status"] == "feasible":
        certificate = result["certificate"]
        assert certificate is not None
        assert "Q" in certificate
        assert "S" in certificate
        assert "multipliers" in certificate
