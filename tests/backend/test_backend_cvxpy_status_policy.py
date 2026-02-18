import pytest

from autolyap import IterationDependent, IterationIndependent, SolverOptions
from autolyap.problemclass import InclusionProblem, MaximallyMonotone


class _FakeCP:
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    INFEASIBLE_INACCURATE = "infeasible_inaccurate"
    UNBOUNDED = "unbounded"
    UNBOUNDED_INACCURATE = "unbounded_inaccurate"


class _FixedStatusProblem:
    def __init__(self, status: str):
        self.status = status

    def solve(self, **kwargs):
        del kwargs
        return None


def _iteration_independent_operator_data(tiny_operator_algorithm):
    problem = InclusionProblem([MaximallyMonotone()])
    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_operator_algorithm
    )
    return problem, P, T


def _iteration_dependent_operator_data(tiny_operator_algorithm):
    problem = InclusionProblem([MaximallyMonotone()])
    Q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=0, i=1, j=1
    )
    Q_1 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=1, i=1, j=1
    )
    return problem, Q_0, Q_1


@pytest.mark.parametrize(
    ("accept_inaccurate", "expected_status"),
    [
        (True, "feasible"),
        (False, "not_solved"),
    ],
)
def test_iteration_independent_search_cvxpy_inaccurate_status_policy(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
    accept_inaccurate: bool,
    expected_status: str,
):
    problem, P, T = _iteration_independent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(IterationIndependent, "_import_cvxpy", staticmethod(lambda: _FakeCP))
    monkeypatch.setattr(
        IterationIndependent,
        "_build_iteration_independent_problem_cvxpy",
        staticmethod(lambda *args, **kwargs: (_FixedStatusProblem(_FakeCP.OPTIMAL_INACCURATE), {})),
    )
    monkeypatch.setattr(
        IterationIndependent,
        "_extract_iteration_independent_certificate_cvxpy",
        staticmethod(lambda _handles: {"mock": "certificate"}),
    )

    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        rho=1.0,
        solver_options=SolverOptions(
            backend="cvxpy",
            cvxpy_accept_inaccurate=accept_inaccurate,
        ),
        verbosity=0,
    )
    assert result["status"] == expected_status
    assert result["solve_status"] == _FakeCP.OPTIMAL_INACCURATE
    if accept_inaccurate:
        assert result["certificate"] == {"mock": "certificate"}
    else:
        assert result["certificate"] is None


@pytest.mark.parametrize(
    ("accept_inaccurate", "expected_status"),
    [
        (True, "feasible"),
        (False, "not_solved"),
    ],
)
def test_iteration_dependent_search_cvxpy_inaccurate_status_policy(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
    accept_inaccurate: bool,
    expected_status: str,
):
    problem, Q_0, Q_1 = _iteration_dependent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(IterationDependent, "_import_cvxpy", staticmethod(lambda: _FakeCP))
    monkeypatch.setattr(
        IterationDependent,
        "_build_iteration_dependent_problem_cvxpy",
        staticmethod(lambda *args, **kwargs: (_FixedStatusProblem(_FakeCP.OPTIMAL_INACCURATE), {"c_K_var": 1.0})),
    )
    monkeypatch.setattr(
        IterationDependent,
        "_extract_scalar_variable_value",
        staticmethod(lambda _var: 1.0),
    )
    monkeypatch.setattr(
        IterationDependent,
        "_extract_iteration_dependent_certificate_cvxpy",
        staticmethod(lambda _handles: {"mock": "certificate"}),
    )

    result = IterationDependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        1,
        Q_0,
        Q_1,
        solver_options=SolverOptions(
            backend="cvxpy",
            cvxpy_accept_inaccurate=accept_inaccurate,
        ),
        verbosity=0,
    )
    assert result["status"] == expected_status
    assert result["solve_status"] == _FakeCP.OPTIMAL_INACCURATE
    if accept_inaccurate:
        assert result["c_K"] == pytest.approx(1.0)
        assert result["certificate"] == {"mock": "certificate"}
    else:
        assert result["c_K"] is None
        assert result["certificate"] is None
