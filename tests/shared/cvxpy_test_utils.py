"""Shared CVXPY/MOSEK test helpers for backend and convergence test modules."""

from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Optional, Sequence, Set, cast

import pytest

from autolyap import SolverOptions
from autolyap.solver_options import _DEFAULT_CVXPY_SOLVER_PARAMS
from tests.shared.mosek_utils import require_mosek_license, skip_or_fail_mosek

_DEFAULT_SDPA_TEST_PARAMS = cast(Mapping[str, Any], _DEFAULT_CVXPY_SOLVER_PARAMS["SDPA"])
_DEFAULT_SDPA_MULTIPRECISION_TEST_PARAMS = {
    "maxIteration": 500,
    "epsilonStar": 1e-30,
    "epsilonDash": 1e-30,
    "mpfPrecision": 512,
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


def require_cvxpy_scs_solver() -> str:
    cp = require_cvxpy_module()
    installed = get_installed_cvxpy_solvers(cp)
    if "SCS" not in installed:
        pytest.skip("CVXPY SCS solver is not available.")
    return "SCS"


def require_cvxpy_copt_solver() -> str:
    cp = require_cvxpy_module()
    installed = get_installed_cvxpy_solvers(cp)
    if "COPT" not in installed:
        pytest.skip("CVXPY COPT solver is not available.")
    return "COPT"


def require_cvxpy_sdpa_solver() -> str:
    cp = require_cvxpy_module()
    installed = get_installed_cvxpy_solvers(cp)
    if "SDPA" not in installed:
        pytest.skip("CVXPY SDPA solver is not available.")
    return "SDPA"


def _is_python_distribution_installed(distribution_name: str) -> bool:
    try:
        importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def require_cvxpy_sdpa_multiprecision_solver() -> str:
    solver_name = require_cvxpy_sdpa_solver()
    if not any(
        _is_python_distribution_installed(name)
        for name in ("sdpa-multiprecision", "sdpa_multiprecision")
    ):
        pytest.skip(
            "SDPA multiprecision backend is not available. "
            "Install it with `pip install sdpa-multiprecision`."
        )
    return solver_name


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
    default_params = _DEFAULT_CVXPY_SOLVER_PARAMS.get(solver_name.upper())
    if isinstance(default_params, Mapping):
        params.update(dict(default_params))
    if extra_params is not None:
        params.update(dict(extra_params))
    return SolverOptions(
        backend="cvxpy",
        cvxpy_solver=solver_name,
        cvxpy_solver_params=params,
        cvxpy_accept_inaccurate=not strict_status,
    )


def make_cvxpy_sdpa_options(
        *,
        multiprecision: bool = False,
        strict_status: bool = False,
        extra_params: Optional[Mapping[str, Any]] = None,
) -> SolverOptions:
    params = dict(_DEFAULT_SDPA_TEST_PARAMS)
    if multiprecision:
        params.update(_DEFAULT_SDPA_MULTIPRECISION_TEST_PARAMS)
    if extra_params is not None:
        params.update(dict(extra_params))
    params.setdefault("verbose", False)
    return SolverOptions(
        backend="cvxpy",
        cvxpy_solver="SDPA",
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
