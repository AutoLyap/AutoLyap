from autolyap import IterationDependent
from autolyap.problemclass import Convex, InclusionProblem, MaximallyMonotone


def test_iteration_dependent_verify_with_cvxpy_backend_smoke(
    tiny_functional_algorithm, cvxpy_open_source_solver_options
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
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"success", "c_K", "certificate"}
    if result["success"]:
        certificate = result["certificate"]
        assert result["c_K"] is not None
        assert certificate is not None
        assert "Q_sequence" in certificate
        assert "q_sequence" in certificate
        assert "multipliers" in certificate


def test_iteration_dependent_verify_with_cvxpy_operator_only_schema(
    tiny_operator_algorithm, cvxpy_open_source_solver_options
):
    problem = InclusionProblem([MaximallyMonotone()])
    Q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=0, i=1, j=1
    )
    Q_1 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=1, i=1, j=1
    )
    result = IterationDependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        1,
        Q_0,
        Q_1,
        solver_options=cvxpy_open_source_solver_options,
    )
    assert set(result.keys()) == {"success", "c_K", "certificate"}
    if result["success"]:
        certificate = result["certificate"]
        assert certificate is not None
        assert certificate["q_sequence"] is None
        assert len(certificate["Q_sequence"]) == 2


def test_iteration_dependent_verify_verbosity_reports_equality_section(
    tiny_functional_algorithm, cvxpy_open_source_solver_options, capsys
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
        solver_options=cvxpy_open_source_solver_options,
        verbosity=1,
    )
    captured = capsys.readouterr()
    assert "Solving iteration-dependent SDP" in captured.out
    if result["success"]:
        assert "Iteration-dependent SDP diagnostics" in captured.out
        assert "Nonnegativity check:" in captured.out
        assert "PSD check:" in captured.out
        assert "Equality check:" in captured.out
    else:
        assert (
            "Iteration-dependent SDP status="
            in captured.out
            or "Iteration-dependent SDP solve failed"
            in captured.out
        )
    assert set(result.keys()) == {"success", "c_K", "certificate"}


def test_iteration_dependent_verify_operator_only_verbosity_reports_no_equalities(
    tiny_operator_algorithm, cvxpy_open_source_solver_options, capsys
):
    problem = InclusionProblem([MaximallyMonotone()])
    Q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=0, i=1, j=1
    )
    Q_1 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=1, i=1, j=1
    )
    result = IterationDependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        1,
        Q_0,
        Q_1,
        solver_options=cvxpy_open_source_solver_options,
        verbosity=1,
    )
    captured = capsys.readouterr()
    assert "Solving iteration-dependent SDP" in captured.out
    if result["success"]:
        assert "Iteration-dependent SDP diagnostics" in captured.out
        assert "Equality check: no active equality constraints." in captured.out
    else:
        assert (
            "Iteration-dependent SDP status="
            in captured.out
            or "Iteration-dependent SDP solve failed"
            in captured.out
        )
    assert set(result.keys()) == {"success", "c_K", "certificate"}
