from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from autolyap.utils.backend_types import CvxpyStatusModuleProtocol


SUPPORTED_SOLVER_BACKENDS = ("mosek_fusion", "cvxpy")
_DEFAULT_MOSEK_FUSION_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}
_DEFAULT_CVXPY_SOLVER_PARAMS = {
    "CLARABEL": {
        "max_iter": 2000,
        "tol_feas": 1e-8,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
    },
    "SCS": {
        "eps": 1e-6,
        "max_iters": 200000,
        "acceleration_lookback": 0,
    },
    "SDPA": {
        "maxIteration": 100,
        "epsilonStar": 1e-7,
        "epsilonDash": 1e-7,
    },
    "COPT": {
        "SDPMethod": 0,
        "BarIterLimit": 500,
        "FeasTol": 1e-7,
        "DualTol": 1e-7,
        "RelGap": 1e-8,
        "AbsGap": 1e-8,
        "Presolve": -1,
        "Scaling": -1,
        "Dualize": -1,
    },
}


@dataclass(frozen=True)
class SolverOptions:
    r"""
    Configure solver backend selection and backend-specific options.

    **Used in**

    - ``solver_options`` argument of
      :meth:`~autolyap.IterationIndependent.search_lyapunov`.
    - ``solver_options`` argument of
      :meth:`~autolyap.IterationIndependent.LinearConvergence.bisection_search_rho`.
    - ``solver_options`` argument of
      :meth:`~autolyap.IterationDependent.search_lyapunov`.

    **Parameters**

    ``backend`` (``str``, default ``"mosek_fusion"``)
        Selects the modeling/solver backend.
        ``"mosek_fusion"`` builds and solves directly with MOSEK Fusion.
        ``"cvxpy"`` builds the problem in CVXPY and solves it with
        ``cvxpy_solver``.

    ``mosek_params`` (``Mapping[str, Any] | None``, default ``None``)
        Extra MOSEK Fusion solver options as key-value pairs.
        Used only when ``backend="mosek_fusion"``.
        If ``None``, AutoLyap applies the explicit default MOSEK profile:
        ``intpntCoTolPfeas=1e-8``, ``intpntCoTolDfeas=1e-8``,
        ``intpntCoTolRelGap=1e-8``, and ``intpntMaxIterations=1000``.

    ``cvxpy_solver`` (``str | None``, default ``None``)
        Name of the CVXPY solver, for example ``"CLARABEL"``, ``"SCS"``,
        ``"MOSEK"``, ``"SDPA"``, or ``"COPT"``.
        If ``None``, CVXPY chooses the solver.
        Used only when ``backend="cvxpy"``.

    ``cvxpy_solver_params`` (``Mapping[str, Any] | None``, default ``None``)
        Extra solver options for the selected CVXPY solver.
        See the examples below for recommended profiles and common options.
        AutoLyap applies ``warm_start=True`` by default.
        Used only when ``backend="cvxpy"``.

    ``cvxpy_accept_inaccurate`` (``bool``, default ``True``)
        Controls which CVXPY statuses are accepted as successful solves.
        If ``True``, accept both ``OPTIMAL`` and ``OPTIMAL_INACCURATE``.
        If ``False``, require ``OPTIMAL``.
        Used only when ``backend="cvxpy"``.

    **Examples**

    .. code-block:: python

       # MOSEK Fusion (explicit default profile; requires `pip install mosek`)
       SolverOptions(
           backend="mosek_fusion",
           mosek_params={
               "intpntCoTolPfeas": 1e-8,  # default
               "intpntCoTolDfeas": 1e-8,  # default
               "intpntCoTolRelGap": 1e-8,  # default
               "intpntMaxIterations": 1000,  # default
           },
       )

       # CVXPY + CLARABEL (explicit default profile)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="CLARABEL",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "max_iter": 2000,  # default
               "tol_feas": 1e-8,  # default
               "tol_gap_abs": 1e-8,  # default
               "tol_gap_rel": 1e-8,  # default
               "warm_start": True,  # default
           },
       )

       # CVXPY + MOSEK (explicit default profile; requires `pip install mosek`)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="MOSEK",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,  # default
               "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,  # default
               "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,  # default
               "warm_start": True,  # default
           },
       )

       # CVXPY + SDPA (explicit default profile; requires `pip install sdpa-python`)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="SDPA",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "maxIteration": 100,  # default
               "epsilonStar": 1e-7,  # default
               "epsilonDash": 1e-7,  # default
               "warm_start": True,  # default
           },
       )

       # CVXPY + SDPA multiprecision (high-precision profile; requires `pip install sdpa-multiprecision`)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="SDPA",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "maxIteration": 500,
               "epsilonStar": 1e-30,
               "epsilonDash": 1e-30,
               "mpfPrecision": 512,
               "warm_start": True,  # default
           },
       )

       # CVXPY + SCS (explicit default profile)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="SCS",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "eps": 1e-6,  # default
               "max_iters": 200000,  # default
               "acceleration_lookback": 0,  # default
               "warm_start": True,  # default
           },
       )

       # CVXPY + COPT (explicit default profile; requires `pip install coptpy`)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="COPT",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "SDPMethod": 0,  # default
               "BarIterLimit": 500,  # default
               "FeasTol": 1e-7,  # default
               "DualTol": 1e-7,  # default
               "RelGap": 1e-8,  # default
               "AbsGap": 1e-8,  # default
               "Presolve": -1,  # default
               "Scaling": -1,  # default
               "Dualize": -1,  # default
               "warm_start": True,  # default
           },
       )

    """

    backend: str = "mosek_fusion"
    mosek_params: Optional[Mapping[str, Any]] = None
    cvxpy_solver: Optional[str] = None
    cvxpy_solver_params: Optional[Mapping[str, Any]] = None
    cvxpy_accept_inaccurate: bool = True


def _normalize_solver_options(options: Optional[SolverOptions]) -> SolverOptions:
    r"""
    Validate and normalize user-provided solver options.

    If `options` is `None`, returns the default ``SolverOptions()`` profile.
    Otherwise, this routine checks backend names, normalizes mapping fields,
    and returns a sanitized immutable `SolverOptions` instance.
    """
    if options is None:
        return SolverOptions()

    if not isinstance(options, SolverOptions):
        raise ValueError(
            "solver_options must be an instance of autolyap.solver_options.SolverOptions or None."
        )

    backend = str(options.backend).strip().lower()
    if backend not in SUPPORTED_SOLVER_BACKENDS:
        raise ValueError(
            f"Unsupported solver backend '{options.backend}'. "
            f"Expected one of {SUPPORTED_SOLVER_BACKENDS}."
        )

    mosek_params = _normalize_named_mapping(options.mosek_params, "mosek_params")
    cvxpy_solver = None
    if options.cvxpy_solver is not None:
        cvxpy_solver = str(options.cvxpy_solver).strip()
        if not cvxpy_solver:
            raise ValueError("cvxpy_solver cannot be an empty string.")
    cvxpy_solver_params = _normalize_cvxpy_solver_params(
        options.cvxpy_solver,
        options.cvxpy_solver_params,
    )
    cvxpy_accept_inaccurate = options.cvxpy_accept_inaccurate
    if not isinstance(cvxpy_accept_inaccurate, bool):
        raise ValueError("cvxpy_accept_inaccurate must be a bool.")

    return SolverOptions(
        backend=backend,
        mosek_params=mosek_params,
        cvxpy_solver=cvxpy_solver,
        cvxpy_solver_params=cvxpy_solver_params,
        cvxpy_accept_inaccurate=cvxpy_accept_inaccurate,
    )


def _get_cvxpy_solve_kwargs(options: SolverOptions) -> Dict[str, Any]:
    r"""
    Build keyword arguments for ``cvxpy.Problem.solve(...)``.

    Merges:
    1. Selected solver name (`solver`).
    2. Built-in defaults for known solvers.
    3. User overrides from ``cvxpy_solver_params``.

    Always defaults ``warm_start=True`` unless explicitly overridden.
    """
    kwargs: Dict[str, Any] = {}
    if options.cvxpy_solver is not None:
        solver_name = options.cvxpy_solver
        kwargs["solver"] = solver_name
        kwargs.update(
            dict(_DEFAULT_CVXPY_SOLVER_PARAMS.get(solver_name.upper(), {}))
        )
    if options.cvxpy_solver_params is not None:
        user_kwargs = dict(options.cvxpy_solver_params)
        if options.cvxpy_solver is not None and options.cvxpy_solver.upper() == "COPT":
            nested_copt_params = user_kwargs.get("params")
            if isinstance(nested_copt_params, Mapping):
                user_kwargs.pop("params")
                flattened_copt_params = dict(nested_copt_params)
                flattened_copt_params.update(user_kwargs)
                user_kwargs = flattened_copt_params
        default_params = kwargs.get("params")
        user_params = user_kwargs.get("params")
        if isinstance(default_params, Mapping) and isinstance(user_params, Mapping):
            merged_params = dict(default_params)
            merged_params.update(dict(user_params))
            user_kwargs["params"] = merged_params
        kwargs.update(user_kwargs)
    kwargs.setdefault("warm_start", True)
    return kwargs


def _get_cvxpy_accepted_statuses(cp: CvxpyStatusModuleProtocol, options: SolverOptions) -> set[str]:
    r"""
    Return CVXPY statuses treated as successful solves.

    Always includes ``OPTIMAL``; optionally includes
    ``OPTIMAL_INACCURATE`` when enabled by solver options.
    """
    statuses = {cp.OPTIMAL}
    if options.cvxpy_accept_inaccurate:
        statuses.add(cp.OPTIMAL_INACCURATE)
    return statuses


def _normalize_mapping(mapping: Optional[Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    r"""Validate an optional mapping and return it as a plain dict copy."""
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping/dict when provided.")
    return dict(mapping)


def _normalize_cvxpy_solver_params(
        cvxpy_solver: Optional[str],
        cvxpy_solver_params: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    r"""
    Validate and normalize CVXPY solver kwargs.

    For ``cvxpy_solver="COPT"``, AutoLyap accepts either a flat parameter map
    (preferred) or ``params={...}`` and flattens the nested mapping.
    For ``cvxpy_solver="MOSEK"``, AutoLyap accepts MOSEK parameters either
    under ``mosek_params={...}`` or as top-level ``MSK_*`` keys.
    """
    normalized = _normalize_mapping(cvxpy_solver_params, "cvxpy_solver_params")
    if normalized is None:
        return None

    solver_name = cvxpy_solver.upper() if cvxpy_solver is not None else None

    if solver_name == "COPT":
        nested_copt_params = normalized.get("params")
        if nested_copt_params is not None:
            if not isinstance(nested_copt_params, Mapping):
                raise ValueError(
                    "For cvxpy_solver='COPT', cvxpy_solver_params['params'] must be a mapping."
                )
            normalized.pop("params")
            flattened_copt_params = dict(nested_copt_params)
            flattened_copt_params.update(normalized)
            normalized = flattened_copt_params

    if solver_name == "MOSEK":
        nested_mosek_params = normalized.get("mosek_params")
        if nested_mosek_params is not None and not isinstance(nested_mosek_params, Mapping):
            raise ValueError(
                "For cvxpy_solver='MOSEK', cvxpy_solver_params['mosek_params'] must be a mapping."
            )

        flat_mosek_params = {
            key: value
            for key, value in normalized.items()
            if _is_mosek_param_key(key)
        }
        for key in flat_mosek_params:
            normalized.pop(key)

        if nested_mosek_params is not None or flat_mosek_params:
            merged_mosek_params: Dict[str, Any] = {}
            if isinstance(nested_mosek_params, Mapping):
                merged_mosek_params.update(dict(nested_mosek_params))
            merged_mosek_params.update(flat_mosek_params)
            normalized["mosek_params"] = merged_mosek_params

    return normalized


def _is_mosek_param_key(key: Any) -> bool:
    r"""Return True when a key looks like a MOSEK string parameter name."""
    return isinstance(key, str) and key.strip().upper().startswith("MSK_")


def _normalize_named_mapping(mapping: Optional[Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    r"""
    Validate an optional mapping with string keys and return a dict copy.

    Keys must be non-empty strings after stripping whitespace.
    """
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping/dict when provided.")
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"All keys in {name} must be non-empty strings.")
        normalized[key] = value
    return normalized
