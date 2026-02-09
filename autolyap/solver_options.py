from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


SUPPORTED_SOLVER_BACKENDS = ("mosek_fusion", "cvxpy")
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
    },
}


@dataclass(frozen=True)
class SolverOptions:
    r"""
    Configure solver backend selection and backend-specific options.

    **Used in**

    - ``solver_options`` argument of
      :meth:`~autolyap.IterationIndependent.verify_iteration_independent_Lyapunov`.
    - ``solver_options`` argument of
      :meth:`~autolyap.IterationIndependent.LinearConvergence.bisection_search_rho`.
    - ``solver_options`` argument of
      :meth:`~autolyap.IterationDependent.verify_iteration_dependent_Lyapunov`.

    **Parameters**

    ``backend`` (``str``, default ``"mosek_fusion"``)
        Backend selector.
        ``"mosek_fusion"`` builds and solves directly with MOSEK Fusion.
        ``"cvxpy"`` builds in CVXPY, then solves with ``cvxpy_solver``.

    ``mosek_params`` (``Mapping[str, Any] | None``, default ``None``)
        MOSEK Fusion parameters forwarded to
        ``Model.setSolverParam(name, value)``.
        Used when ``backend="mosek_fusion"``.

    ``cvxpy_solver`` (``str | None``, default ``None``)
        CVXPY solver name, for example ``"CLARABEL"``, ``"SCS"``, or
        ``"MOSEK"``. If ``None``, CVXPY chooses.
        Used when ``backend="cvxpy"``.

    ``cvxpy_solver_params`` (``Mapping[str, Any] | None``, default ``None``)
        Keyword arguments forwarded to ``cvxpy.Problem.solve(...)``.
        Used when ``backend="cvxpy"``.

    ``cvxpy_accept_inaccurate`` (``bool``, default ``True``)
        If ``True``, accept both ``OPTIMAL`` and ``OPTIMAL_INACCURATE``.
        If ``False``, require ``OPTIMAL``.
        Used when ``backend="cvxpy"``.

    **Examples**

    .. code-block:: python

       # MOSEK Fusion (explicit default profile)
       SolverOptions(
           backend="mosek_fusion",
           mosek_params={
               "intpntCoTolPfeas": 1e-8,   # default
               "intpntCoTolDfeas": 1e-8,   # default
               "intpntCoTolRelGap": 1e-8,  # default
               "intpntMaxIterations": 400, # default
           },
       )

       # CVXPY + CLARABEL (explicit default profile)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="CLARABEL",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "max_iter": 2000,     # default
               "tol_feas": 1e-8,     # default
               "tol_gap_abs": 1e-8,  # default
               "tol_gap_rel": 1e-8,  # default
           },
       )

       # CVXPY + SCS (explicit default profile)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="SCS",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "eps": 1e-6,          # default
               "max_iters": 200000,  # default
           },
       )

       # CVXPY + MOSEK (explicit default profile)
       SolverOptions(
           backend="cvxpy",
           cvxpy_solver="MOSEK",
           cvxpy_accept_inaccurate=True,  # default
           cvxpy_solver_params={
               "mosek_params": {
                   "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,    # default
                   "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,    # default
                   "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,  # default
               },
           },
       )

    **Note**

    If tolerance values go down, increase the corresponding maximum-iteration
    setting.
    """

    backend: str = "mosek_fusion"
    mosek_params: Optional[Mapping[str, Any]] = None
    cvxpy_solver: Optional[str] = None
    cvxpy_solver_params: Optional[Mapping[str, Any]] = None
    cvxpy_accept_inaccurate: bool = True


def normalize_solver_options(options: Optional[SolverOptions]) -> SolverOptions:
    r"""Validate and normalize solver options, filling defaults when omitted."""
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
    cvxpy_solver_params = _normalize_mapping(options.cvxpy_solver_params, "cvxpy_solver_params")
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


def get_cvxpy_solve_kwargs(options: SolverOptions) -> Dict[str, Any]:
    r"""Build keyword arguments for `cvxpy.Problem.solve(...)`."""
    kwargs: Dict[str, Any] = {}
    if options.cvxpy_solver is not None:
        solver_name = options.cvxpy_solver
        kwargs["solver"] = solver_name
        kwargs.update(
            dict(_DEFAULT_CVXPY_SOLVER_PARAMS.get(solver_name.upper(), {}))
        )
    if options.cvxpy_solver_params is not None:
        kwargs.update(dict(options.cvxpy_solver_params))
    kwargs.setdefault("warm_start", True)
    return kwargs


def get_cvxpy_accepted_statuses(cp, options: SolverOptions) -> set:
    r"""Return accepted CVXPY solve statuses for the given options."""
    statuses = {cp.OPTIMAL}
    if options.cvxpy_accept_inaccurate:
        statuses.add(cp.OPTIMAL_INACCURATE)
    return statuses


def _normalize_mapping(mapping: Optional[Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping/dict when provided.")
    return dict(mapping)


def _normalize_named_mapping(mapping: Optional[Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
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
