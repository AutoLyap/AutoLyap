import numpy as np

from autolyap import IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem, MaximallyMonotone
from tests.shared.cvxpy_fixtures import cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
from tests.shared.cvxpy_test_utils import make_cvxpy_solver_options


def test_iteration_independent_verify_with_cvxpy_backend_smoke(
    tiny_functional_algorithm, cvxpy_open_source_solver_options
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    result = IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p=p,
        t=t,
        rho=1.0,
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"success", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["success"]:
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
    assert set(result.keys()) == {"success", "rho", "certificate"}
    if result["success"]:
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
    assert set(result.keys()) == {"success", "rho", "certificate"}


def test_iteration_independent_verify_with_cvxpy_operator_only_schema(
    tiny_operator_algorithm, cvxpy_open_source_solver_options
):
    problem = InclusionProblem([MaximallyMonotone()])
    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_operator_algorithm
    )
    result = IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        rho=1.0,
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"success", "rho", "certificate"}
    assert np.isclose(result["rho"], 1.0)
    if result["success"]:
        certificate = result["certificate"]
        assert certificate is not None
        assert certificate["q"] is None
        assert certificate["s"] is None
        assert certificate["Q"].shape[0] == certificate["Q"].shape[1]
        assert certificate["S"].shape[0] == certificate["S"].shape[1]
