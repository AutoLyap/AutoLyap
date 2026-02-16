import numpy as np

from autolyap import IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem, MaximallyMonotone
from tests.shared.cvxpy_test_utils import make_cvxpy_solver_options


def test_iteration_independent_verify_with_cvxpy_backend_smoke(
    tiny_functional_algorithm, cvxpy_open_source_solver_options
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
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["status"] == "feasible":
        assert result["certificate"] is not None
        assert "Q" in result["certificate"]
        assert "S" in result["certificate"]
        assert "multipliers" in result["certificate"]


def test_iteration_independent_bisection_with_cvxpy_backend_smoke(
    tiny_functional_algorithm, cvxpy_open_source_solver_options
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        lower_bound=1.0,
        upper_bound=1.0,
        tol=1e-8,
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
    if result["status"] == "feasible":
        assert result["rho"] is not None


def test_iteration_independent_bisection_with_cvxpy_warm_start_override(
    tiny_functional_algorithm, cvxpy_open_source_solver_name
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        lower_bound=1.0,
        upper_bound=1.0,
        tol=1e-8,
        solver_options=make_cvxpy_solver_options(
            cvxpy_open_source_solver_name,
            extra_params={"warm_start": False},
        ),
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}


def test_iteration_independent_verify_with_cvxpy_operator_only_schema(
    tiny_operator_algorithm, cvxpy_open_source_solver_options
):
    problem = InclusionProblem([MaximallyMonotone()])
    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_operator_algorithm
    )
    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        rho=1.0,
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["status"] == "feasible":
        certificate = result["certificate"]
        assert certificate is not None
        assert certificate["q"] is None
        assert certificate["s"] is None
        assert certificate["Q"].shape[0] == certificate["Q"].shape[1]
        assert certificate["S"].shape[0] == certificate["S"].shape[1]


def test_iteration_independent_verify_verbosity_reports_equality_section(
    tiny_functional_algorithm, cvxpy_open_source_solver_options, capsys
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
        solver_options=cvxpy_open_source_solver_options,
        verbosity=1,
    )
    captured = capsys.readouterr()
    assert "Solving iteration-independent SDP" in captured.out
    if result["status"] == "feasible":
        assert "Iteration-independent SDP diagnostics" in captured.out
        assert "Nonnegativity check:" in captured.out
        assert "PSD check:" in captured.out
        assert "Equality check:" in captured.out
    else:
        assert (
            "Iteration-independent SDP status="
            in captured.out
            or "Iteration-independent SDP solve failed"
            in captured.out
        )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}


def test_iteration_independent_bisection_verbosity_reports_search_and_equality(
    tiny_functional_algorithm, cvxpy_open_source_solver_options, capsys
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        lower_bound=1.0,
        upper_bound=1.0,
        tol=1e-8,
        solver_options=cvxpy_open_source_solver_options,
        verbosity=1,
    )
    captured = capsys.readouterr()
    assert "Starting rho bisection" in captured.out
    if result["status"] == "feasible":
        assert "Bisection succeeded" in captured.out
        assert "Iteration-independent SDP diagnostics" in captured.out
        assert "Equality check:" in captured.out
    else:
        assert (
            "Bisection aborted" in captured.out
            or "Bisection finished without a feasible terminal rho." in captured.out
        )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}


def test_iteration_independent_verify_operator_only_verbosity_reports_no_equalities(
    tiny_operator_algorithm, cvxpy_open_source_solver_options, capsys
):
    problem = InclusionProblem([MaximallyMonotone()])
    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_operator_algorithm
    )
    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        rho=1.0,
        solver_options=cvxpy_open_source_solver_options,
        verbosity=1,
    )
    captured = capsys.readouterr()
    assert "Solving iteration-independent SDP" in captured.out
    if result["status"] == "feasible":
        assert "Iteration-independent SDP diagnostics" in captured.out
        assert "Equality check: no active equality constraints." in captured.out
    else:
        assert (
            "Iteration-independent SDP status="
            in captured.out
            or "Iteration-independent SDP solve failed"
            in captured.out
        )
    assert set(result.keys()) == {"status", "solve_status", "rho", "certificate"}
