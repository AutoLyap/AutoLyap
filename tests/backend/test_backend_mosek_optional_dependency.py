import builtins

import pytest

from autolyap import IterationIndependent, SolverOptions
from autolyap.problemclass import Convex, InclusionProblem


def _block_mosek_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mosek" or name.startswith("mosek."):
            raise ModuleNotFoundError("blocked mosek import in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_iteration_independent_cvxpy_backend_does_not_require_mosek_import(
    monkeypatch: pytest.MonkeyPatch,
    tiny_functional_algorithm,
    cvxpy_open_source_solver_options,
):
    _block_mosek_import(monkeypatch)

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


def test_iteration_independent_mosek_backend_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tiny_functional_algorithm,
):
    _block_mosek_import(monkeypatch)

    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )

    with pytest.raises(ImportError, match=r"autolyap\[mosek\]"):
        IterationIndependent.search_lyapunov(
            problem,
            tiny_functional_algorithm,
            P,
            T,
            p=p,
            t=t,
            rho=1.0,
            solver_options=SolverOptions(backend="mosek_fusion"),
        )
