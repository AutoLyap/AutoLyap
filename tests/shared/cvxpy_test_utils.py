"""Shared CVXPY/MOSEK test helpers for backend and convergence test modules."""

from typing import Any, Mapping, Optional, Sequence, Set

import pytest

from autolyap import SolverOptions
from tests.shared.mosek_utils import require_mosek_license, skip_or_fail_mosek

_DEFAULT_SCS_TEST_PARAMS = {
    # Tuned to prefer stable OPTIMAL status in CI over aggressively tight accuracy.
    "eps": 1e-5,
    "max_iters": 50000,
    "acceleration_lookback": 0,
}
_DEFAULT_MOSEK_FUSION_PARAMS = {
    "intpntCoTolPfeas": 1e-9,
    "intpntCoTolDfeas": 1e-9,
    "intpntCoTolRelGap": 1e-9,
}
_DEFAULT_CVXPY_MOSEK_PARAMS = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
}


def require_cvxpy_module():
    return pytest.importorskip("cvxpy")


def get_installed_cvxpy_solvers(cp_module) -> Set[str]:
    return set(cp_module.installed_solvers())


def choose_cvxpy_solver(installed_solvers: Set[str], preferred_order: Sequence[str]) -> str:
    for solver_name in preferred_order:
        if solver_name in installed_solvers:
            return solver_name
    raise RuntimeError(f"No preferred solver found. Installed solvers: {sorted(installed_solvers)}")


def require_open_source_cvxpy_solver(preferred_order: Sequence[str] = ("CLARABEL", "SCS")) -> str:
    cp = require_cvxpy_module()
    installed = get_installed_cvxpy_solvers(cp)
    try:
        return choose_cvxpy_solver(installed, preferred_order)
    except RuntimeError:
        preferred = "/".join(preferred_order)
        pytest.skip(f"CVXPY is installed, but no supported SDP solver ({preferred}) is available.")


def require_cvxpy_clarabel_solver() -> str:
    cp = require_cvxpy_module()
    installed = get_installed_cvxpy_solvers(cp)
    if "CLARABEL" not in installed:
        pytest.skip("CVXPY CLARABEL solver is not available.")
    return "CLARABEL"


def require_cvxpy_mosek_solver() -> str:
    try:
        import cvxpy as cp
    except ModuleNotFoundError:
        skip_or_fail_mosek("CVXPY is not installed.")
    installed = get_installed_cvxpy_solvers(cp)
    if "MOSEK" not in installed:
        skip_or_fail_mosek("CVXPY MOSEK solver interface is not available.")
    require_mosek_license()
    return "MOSEK"


def make_cvxpy_solver_options(
        solver_name: str,
        *,
        strict_status: bool = False,
        extra_params: Optional[Mapping[str, Any]] = None,
) -> SolverOptions:
    params = {"verbose": False}
    if solver_name == "SCS":
        params.update(_DEFAULT_SCS_TEST_PARAMS)
    if extra_params is not None:
        params.update(dict(extra_params))
    return SolverOptions(
        backend="cvxpy",
        cvxpy_solver=solver_name,
        cvxpy_solver_params=params,
        cvxpy_accept_inaccurate=not strict_status,
    )


def make_cvxpy_mosek_options(
        *,
        strict_status: bool = True,
        mosek_params: Optional[Mapping[str, Any]] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
) -> SolverOptions:
    params = {"verbose": False}
    params["mosek_params"] = dict(_DEFAULT_CVXPY_MOSEK_PARAMS if mosek_params is None else mosek_params)
    if extra_params is not None:
        params.update(dict(extra_params))
    return SolverOptions(
        backend="cvxpy",
        cvxpy_solver="MOSEK",
        cvxpy_solver_params=params,
        cvxpy_accept_inaccurate=not strict_status,
    )


def make_mosek_fusion_options(
        mosek_params: Optional[Mapping[str, Any]] = None,
) -> SolverOptions:
    return SolverOptions(
        backend="mosek_fusion",
        mosek_params=dict(_DEFAULT_MOSEK_FUSION_PARAMS if mosek_params is None else mosek_params),
    )
