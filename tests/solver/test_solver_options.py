import pytest

from autolyap import SolverOptions
from autolyap.solver_options import (
    SUPPORTED_SOLVER_BACKENDS,
    _get_cvxpy_accepted_statuses,
    _get_cvxpy_solve_kwargs,
    _normalize_solver_options,
)


def test_solver_options_defaults_to_mosek_fusion():
    options = _normalize_solver_options(None)
    assert options.backend == "mosek_fusion"
    assert options.mosek_params is None
    assert options.cvxpy_solver is None
    assert options.cvxpy_solver_params is None
    assert options.cvxpy_accept_inaccurate is True


def test_solver_options_rejects_unknown_backend():
    with pytest.raises(ValueError):
        _normalize_solver_options(SolverOptions(backend="unknown"))


def test_solver_options_rejects_non_string_mosek_param_keys():
    with pytest.raises(ValueError):
        _normalize_solver_options(
            SolverOptions(mosek_params={1: 1e-8})  # type: ignore[dict-item]
        )


def test_solver_options_normalizes_backend_case():
    options = _normalize_solver_options(SolverOptions(backend="CVXPY"))
    assert options.backend == "cvxpy"
    assert options.backend in SUPPORTED_SOLVER_BACKENDS


def test_get_cvxpy_solve_kwargs_merges_solver_and_params():
    options = _normalize_solver_options(
        SolverOptions(
            backend="cvxpy",
            cvxpy_solver="SCS",
            cvxpy_solver_params={"max_iters": 123},
        )
    )
    kwargs = _get_cvxpy_solve_kwargs(options)
    assert kwargs["solver"] == "SCS"
    assert kwargs["max_iters"] == 123
    assert kwargs["eps"] == 1e-6
    assert kwargs["warm_start"] is True


def test_get_cvxpy_solve_kwargs_allows_warm_start_override():
    options = _normalize_solver_options(
        SolverOptions(
            backend="cvxpy",
            cvxpy_solver_params={"warm_start": False},
        )
    )
    kwargs = _get_cvxpy_solve_kwargs(options)
    assert kwargs["warm_start"] is False


def test_get_cvxpy_solve_kwargs_applies_clarabel_defaults():
    options = _normalize_solver_options(
        SolverOptions(
            backend="cvxpy",
            cvxpy_solver="CLARABEL",
        )
    )
    kwargs = _get_cvxpy_solve_kwargs(options)
    assert kwargs["solver"] == "CLARABEL"
    assert kwargs["max_iter"] == 2000
    assert kwargs["tol_feas"] == 1e-8
    assert kwargs["tol_gap_abs"] == 1e-8
    assert kwargs["tol_gap_rel"] == 1e-8
    assert kwargs["warm_start"] is True


def test_get_cvxpy_solve_kwargs_user_params_override_defaults():
    options = _normalize_solver_options(
        SolverOptions(
            backend="cvxpy",
            cvxpy_solver="CLARABEL",
            cvxpy_solver_params={"max_iter": 2500, "tol_feas": 5e-8},
        )
    )
    kwargs = _get_cvxpy_solve_kwargs(options)
    assert kwargs["max_iter"] == 2500
    assert kwargs["tol_feas"] == 5e-8
    assert kwargs["tol_gap_abs"] == 1e-8
    assert kwargs["tol_gap_rel"] == 1e-8


def test_solver_options_rejects_non_bool_cvxpy_accept_inaccurate():
    with pytest.raises(ValueError):
        _normalize_solver_options(
            SolverOptions(cvxpy_accept_inaccurate="yes")  # type: ignore[arg-type]
        )


def test_get_cvxpy_accepted_statuses_default_includes_inaccurate():
    class _CP:
        OPTIMAL = "optimal"
        OPTIMAL_INACCURATE = "optimal_inaccurate"

    statuses = _get_cvxpy_accepted_statuses(_CP, _normalize_solver_options(None))
    assert statuses == {"optimal", "optimal_inaccurate"}


def test_get_cvxpy_accepted_statuses_strict_excludes_inaccurate():
    class _CP:
        OPTIMAL = "optimal"
        OPTIMAL_INACCURATE = "optimal_inaccurate"

    options = _normalize_solver_options(
        SolverOptions(backend="cvxpy", cvxpy_accept_inaccurate=False)
    )
    statuses = _get_cvxpy_accepted_statuses(_CP, options)
    assert statuses == {"optimal"}
