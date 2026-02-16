import pytest

from autolyap import IterationDependent, IterationIndependent, SolverOptions
from autolyap.problemclass import InclusionProblem, MaximallyMonotone


class _FakeCP:
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"

    @staticmethod
    def Parameter(*, nonneg=True, value=None):
        del nonneg
        return _FakeCPParameter(value=value)


class _FakeCPParameter:
    def __init__(self, value=None):
        self.value = value


class _RaisingProblem:
    def __init__(self, message: str):
        self._message = message
        self.status = None

    def solve(self, **kwargs):
        raise RuntimeError(self._message)


class _FakeOptimizeError(Exception):
    pass


class _FakeRhoParam:
    def index(self, _index: int):
        return object()

    def setValue(self, _value):
        return None


class _FakeMosekModel:
    def parameter(self, _size: int):
        return _FakeRhoParam()

    def solve(self):
        raise _FakeOptimizeError("internal mosek failure")

    def dispose(self):
        return None


class _FakeMosekModule:
    OptimizeError = _FakeOptimizeError
    Model = _FakeMosekModel


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


def test_iteration_independent_search_cvxpy_solver_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, P, T = _iteration_independent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(IterationIndependent, "_import_cvxpy", staticmethod(lambda: _FakeCP))
    monkeypatch.setattr(
        IterationIndependent,
        "_build_iteration_independent_problem_cvxpy",
        staticmethod(lambda *args, **kwargs: (_RaisingProblem("cvxpy-search-boom"), {})),
    )

    with pytest.raises(RuntimeError, match="cvxpy-search-boom"):
        IterationIndependent.search_lyapunov(
            problem,
            tiny_operator_algorithm,
            P,
            T,
            rho=1.0,
            solver_options=SolverOptions(backend="cvxpy"),
        )


def test_iteration_independent_bisection_cvxpy_solver_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, P, T = _iteration_independent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(IterationIndependent, "_import_cvxpy", staticmethod(lambda: _FakeCP))
    monkeypatch.setattr(
        IterationIndependent,
        "_build_iteration_independent_problem_cvxpy",
        staticmethod(lambda *args, **kwargs: (_RaisingProblem("cvxpy-bisection-boom"), {})),
    )

    with pytest.raises(RuntimeError, match="cvxpy-bisection-boom"):
        IterationIndependent.LinearConvergence.bisection_search_rho(
            problem,
            tiny_operator_algorithm,
            P,
            T,
            lower_bound=1.0,
            upper_bound=1.0,
            tol=1e-12,
            solver_options=SolverOptions(backend="cvxpy"),
        )


def test_iteration_independent_search_mosek_optimize_error_returns_not_solved(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, P, T = _iteration_independent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(
        IterationIndependent, "_import_mosek_fusion", staticmethod(lambda: _FakeMosekModule)
    )
    monkeypatch.setattr(
        IterationIndependent,
        "_build_iteration_independent_model",
        staticmethod(lambda *args, **kwargs: (_FakeMosekModel(), {})),
    )
    monkeypatch.setattr(
        IterationIndependent, "_apply_mosek_solver_params", staticmethod(lambda *args, **kwargs: None)
    )

    result = IterationIndependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        rho=1.0,
        solver_options=SolverOptions(backend="mosek_fusion"),
    )
    assert result["status"] != "feasible"
    assert result["status"] == "not_solved"
    assert result["solve_status"] == "optimize_error"
    assert result["certificate"] is None


def test_iteration_independent_bisection_mosek_optimize_error_returns_not_solved(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, P, T = _iteration_independent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(
        IterationIndependent, "_import_mosek_fusion", staticmethod(lambda: _FakeMosekModule)
    )
    monkeypatch.setattr(
        IterationIndependent,
        "_build_iteration_independent_model",
        staticmethod(lambda *args, **kwargs: (kwargs["model"], {})),
    )
    monkeypatch.setattr(
        IterationIndependent, "_apply_mosek_solver_params", staticmethod(lambda *args, **kwargs: None)
    )

    result = IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        tiny_operator_algorithm,
        P,
        T,
        lower_bound=1.0,
        upper_bound=1.0,
        tol=1e-12,
        solver_options=SolverOptions(backend="mosek_fusion"),
    )
    assert result["status"] != "feasible"
    assert result["status"] == "not_solved"
    assert result["solve_status"] == "optimize_error"
    assert result["certificate"] is None


def test_iteration_dependent_search_cvxpy_solver_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, Q_0, Q_1 = _iteration_dependent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(IterationDependent, "_import_cvxpy", staticmethod(lambda: _FakeCP))
    monkeypatch.setattr(
        IterationDependent,
        "_build_iteration_dependent_problem_cvxpy",
        staticmethod(lambda *args, **kwargs: (_RaisingProblem("cvxpy-dependent-boom"), {})),
    )

    with pytest.raises(RuntimeError, match="cvxpy-dependent-boom"):
        IterationDependent.search_lyapunov(
            problem,
            tiny_operator_algorithm,
            1,
            Q_0,
            Q_1,
            solver_options=SolverOptions(backend="cvxpy"),
        )


def test_iteration_dependent_search_mosek_optimize_error_returns_not_solved(
    monkeypatch: pytest.MonkeyPatch,
    tiny_operator_algorithm,
):
    problem, Q_0, Q_1 = _iteration_dependent_operator_data(tiny_operator_algorithm)
    monkeypatch.setattr(
        IterationDependent, "_import_mosek_fusion", staticmethod(lambda: _FakeMosekModule)
    )
    monkeypatch.setattr(
        IterationDependent,
        "_build_iteration_dependent_model",
        staticmethod(lambda *args, **kwargs: (kwargs["model"], {"c_K_var": object()})),
    )
    monkeypatch.setattr(
        IterationDependent, "_apply_mosek_solver_params", staticmethod(lambda *args, **kwargs: None)
    )

    result = IterationDependent.search_lyapunov(
        problem,
        tiny_operator_algorithm,
        1,
        Q_0,
        Q_1,
        solver_options=SolverOptions(backend="mosek_fusion"),
    )
    assert result["status"] != "feasible"
    assert result["status"] == "not_solved"
    assert result["solve_status"] == "optimize_error"
    assert result["certificate"] is None
