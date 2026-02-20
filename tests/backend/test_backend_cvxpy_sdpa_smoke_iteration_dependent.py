# Smoke checks for CVXPY SDPA backend on iteration-dependent Lyapunov searches.

from autolyap import IterationDependent
from autolyap.problemclass import Convex, InclusionProblem


def test_iteration_dependent_verify_with_cvxpy_sdpa_backend_smoke(
    tiny_functional_algorithm, cvxpy_sdpa_solver_options
):
    problem = InclusionProblem([Convex()])
    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_functional_algorithm, k=0, i=1, j=1
    )
    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
        tiny_functional_algorithm, k=1, j=1
    )
    result = IterationDependent.search_lyapunov(
        problem,
        tiny_functional_algorithm,
        1,
        Q_0,
        Q_K,
        q_0=q_0,
        q_K=q_K,
        solver_options=cvxpy_sdpa_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "c_K", "certificate"}
    if result["status"] == "feasible":
        certificate = result["certificate"]
        assert result["c_K"] is not None
        assert certificate is not None
        assert "Q_sequence" in certificate
        assert "q_sequence" in certificate
        assert "multipliers" in certificate


def test_iteration_dependent_verify_with_cvxpy_sdpa_multiprecision_backend_smoke(
    tiny_functional_algorithm, cvxpy_sdpa_multiprecision_solver_options
):
    assert cvxpy_sdpa_multiprecision_solver_options.cvxpy_solver_params is not None
    assert cvxpy_sdpa_multiprecision_solver_options.cvxpy_solver_params["mpfPrecision"] == 512

    problem = InclusionProblem([Convex()])
    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_functional_algorithm, k=0, i=1, j=1
    )
    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
        tiny_functional_algorithm, k=1, j=1
    )
    result = IterationDependent.search_lyapunov(
        problem,
        tiny_functional_algorithm,
        1,
        Q_0,
        Q_K,
        q_0=q_0,
        q_K=q_K,
        solver_options=cvxpy_sdpa_multiprecision_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "c_K", "certificate"}
    if result["status"] == "feasible":
        certificate = result["certificate"]
        assert result["c_K"] is not None
        assert certificate is not None
        assert "Q_sequence" in certificate
        assert "q_sequence" in certificate
        assert "multipliers" in certificate
