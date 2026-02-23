import numpy as np
from typing import Any, Optional, Tuple, Union, List, Dict, Iterator, Mapping, NoReturn, TypedDict, cast
from itertools import combinations
from autolyap.utils.helper_functions import create_symmetric_matrix_expression, create_symmetric_matrix
from autolyap.utils.backend_types import (
    CvxpyModuleProtocol,
    CvxpyStatusModuleProtocol,
    CvxpyValueHandleProtocol,
    MosekFusionModuleProtocol,
    MosekLevelHandleProtocol,
    MosekModelProtocol,
    MosekUpperTriangleSolutionHandleProtocol,
    ScalarVariableHandle,
    SupportsStringConversion,
)
from autolyap.solver_options import (
    SolverOptions,
    _DEFAULT_MOSEK_FUSION_PARAMS,
    _normalize_solver_options,
    _get_cvxpy_solve_kwargs,
    _get_cvxpy_accepted_statuses,
)
from autolyap.utils.validation import (
    ensure_finite_array,
    ensure_integral,
)
from autolyap.problemclass import InclusionProblem
from autolyap.algorithms import Algorithm

Pair = Union[Tuple[int, int], Tuple[str, str]]
PairTuple = Tuple[Pair, ...]
OperatorInterpolationData = Tuple[np.ndarray, Any]
FunctionInterpolationData = Tuple[np.ndarray, np.ndarray, bool, Any]
InterpolationData = Union[OperatorInterpolationData, FunctionInterpolationData]

IterationDependentMultiplierKey = Tuple[int, int, PairTuple, int]
IterationDependentMultiplierMap = Dict[IterationDependentMultiplierKey, ScalarVariableHandle]


class _ReadablePair(TypedDict):
    j: Union[int, str]
    k: Union[int, str]


class _IterationDependentMultiplierRecord(TypedDict):
    iteration: int
    component: int
    interpolation_index: int
    pairs: List[_ReadablePair]
    value: float


class _IterationDependentMultipliers(TypedDict):
    operator_lambda: List[_IterationDependentMultiplierRecord]
    function_lambda: List[_IterationDependentMultiplierRecord]
    function_nu: List[_IterationDependentMultiplierRecord]


class _IterationDependentCertificate(TypedDict):
    Q_sequence: List[np.ndarray]
    q_sequence: Optional[List[np.ndarray]]
    multipliers: _IterationDependentMultipliers


class _IterationDependentResult(TypedDict):
    status: str
    solve_status: Optional[str]
    c_K: Optional[float]
    certificate: Optional[_IterationDependentCertificate]


class _IterationDependentMosekSolutionHandles(TypedDict):
    K: int
    dim_Q: int
    dim_q: Optional[int]
    m_func: int
    Q_0: np.ndarray
    Q_K: np.ndarray
    q_0: Optional[np.ndarray]
    q_K: Optional[np.ndarray]
    Qij_vars: Dict[int, MosekUpperTriangleSolutionHandleProtocol]
    q_vars: Dict[int, MosekLevelHandleProtocol]
    lambdas_op: IterationDependentMultiplierMap
    lambdas_func: IterationDependentMultiplierMap
    nus_func: IterationDependentMultiplierMap
    c_K_var: ScalarVariableHandle


class _IterationDependentCvxpySolutionHandles(TypedDict):
    K: int
    dim_Q: int
    dim_q: Optional[int]
    m_func: int
    Q_0: np.ndarray
    Q_K: np.ndarray
    q_0: Optional[np.ndarray]
    q_K: Optional[np.ndarray]
    Q_vars: Dict[int, CvxpyValueHandleProtocol]
    q_vars: Dict[int, CvxpyValueHandleProtocol]
    lambdas_op: IterationDependentMultiplierMap
    lambdas_func: IterationDependentMultiplierMap
    nus_func: IterationDependentMultiplierMap
    c_K_var: ScalarVariableHandle

_MOSEK_LICENSE_ERROR_MARKERS = (
    "err_license_expired",
    "err_license_max",
    "err_license_server",
    "err_missing_license_file",
)

_RESULT_STATUS_FEASIBLE = "feasible"
_RESULT_STATUS_INFEASIBLE = "infeasible"
_RESULT_STATUS_NOT_SOLVED = "not_solved"


def _is_mosek_license_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return any(marker in error_text for marker in _MOSEK_LICENSE_ERROR_MARKERS)


def _normalize_mosek_status(status: SupportsStringConversion) -> str:
    return "".join(ch for ch in str(status).lower() if ch.isalnum())


def _is_mosek_primal_feasible_status(status: SupportsStringConversion) -> bool:
    normalized = _normalize_mosek_status(status)
    return (
        "primalanddualfeasible" in normalized
        or ("primalfeasible" in normalized and "infeasible" not in normalized)
    )


def _classify_mosek_problem_status(status: SupportsStringConversion) -> str:
    if _is_mosek_primal_feasible_status(status):
        return _RESULT_STATUS_FEASIBLE
    normalized = _normalize_mosek_status(status)
    if "infeasible" in normalized:
        return _RESULT_STATUS_INFEASIBLE
    return _RESULT_STATUS_NOT_SOLVED


def _classify_cvxpy_problem_status(
    status: str,
    cp: CvxpyStatusModuleProtocol,
    accepted_statuses: set[str],
) -> str:
    if status in accepted_statuses:
        return _RESULT_STATUS_FEASIBLE

    infeasible_statuses = {
        getattr(cp, "INFEASIBLE", None),
        getattr(cp, "INFEASIBLE_INACCURATE", None),
        getattr(cp, "UNBOUNDED", None),
        getattr(cp, "UNBOUNDED_INACCURATE", None),
    }
    if status in infeasible_statuses:
        return _RESULT_STATUS_INFEASIBLE
    return _RESULT_STATUS_NOT_SOLVED


def _make_iteration_dependent_result(
    status: str,
    solve_status: Optional[str],
    c_K: Optional[float],
    certificate: Optional[_IterationDependentCertificate],
) -> _IterationDependentResult:
    return {
        "status": status,
        "solve_status": solve_status,
        "c_K": c_K,
        "certificate": certificate,
    }


class _IterationDependentMeta(type):
    def __getattr__(cls, name: str) -> NoReturn:
        if name == "verify_iteration_dependent_Lyapunov":
            raise AttributeError(
                "IterationDependent.verify_iteration_dependent_Lyapunov was removed in v0.2.0. "
                "Use IterationDependent.search_lyapunov instead. "
                "Migration: https://autolyap.github.io/release_notes/v0_2_0.html. "
                "Quick start: https://autolyap.github.io/quick_start.html."
            )
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


class IterationDependent(metaclass=_IterationDependentMeta):
    r"""
    Iteration-dependent Lyapunov analysis utilities.

    For the mathematical formulation, notation, and convergence statements,
    see :doc:`/theory/iteration_dependent_analyses`.

    This class provides the corresponding computational interface, with
    :meth:`search_lyapunov` as the main entry point.
    """
    @staticmethod
    def _import_cvxpy() -> CvxpyModuleProtocol:
        r"""
        Import CVXPY lazily for the optional CVXPY backend.

        **Parameters**

        - `None`.

        **Returns**

        - Module: The imported `cvxpy` module.

        **Raises**

        - `ImportError`: If CVXPY is not installed.
        """
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "CVXPY backend requested, but cvxpy is not installed. "
                "Install it with `pip install cvxpy`."
            ) from exc
        return cp

    @staticmethod
    def _import_mosek_fusion() -> MosekFusionModuleProtocol:
        r"""
        Import MOSEK Fusion lazily for the optional MOSEK backend.

        **Parameters**

        - `None`.

        **Returns**

        - Module: The imported `mosek.fusion` module.

        **Raises**

        - `ImportError`: If MOSEK is not installed.
        """
        try:
            import mosek.fusion as mf
            import mosek.fusion.pythonic  # noqa: F401  # required for Fusion operator overloads
        except ImportError as exc:
            raise ImportError(
                "MOSEK Fusion backend requested, but `mosek` is not installed. "
                "Install it with `pip install autolyap[mosek]`."
            ) from exc
        return mf

    @staticmethod
    def _apply_mosek_solver_params(mod: MosekModelProtocol, solver_options: SolverOptions) -> None:
        r"""
        Apply user-provided MOSEK Fusion parameters to `mod`.

        **Parameters**

        - `mod` (:class:`mosek.fusion.Model`): Target Fusion model.
        - `solver_options` (:class:`~autolyap.solver_options.SolverOptions`): Solver option container.

        **Returns**

        - `None`: Parameters are applied in place.
        """
        params = dict(_DEFAULT_MOSEK_FUSION_PARAMS)
        if solver_options.mosek_params is not None:
            params.update(solver_options.mosek_params)
        for name, value in params.items():
            mod.setSolverParam(name, value)

    @staticmethod
    def _extract_scalar_variable_value(var: ScalarVariableHandle) -> float:
        r"""
        Extract a scalar value from a backend variable handle.

        Supports Fusion handles exposing `level()` and CVXPY handles exposing `value`.

        **Parameters**

        - `var` (:class:`~typing.Any`): Backend scalar variable/expression handle.

        **Returns**

        - (:class:`float`): Extracted scalar value.

        **Raises**

        - `ValueError`: If `var` is not a supported backend handle.
        """
        if hasattr(var, "level"):
            value_arr = np.asarray(var.level(), dtype=float).reshape(-1)
        elif hasattr(var, "value"):
            value_arr = np.asarray(var.value, dtype=float).reshape(-1)
        else:
            raise ValueError("Unsupported scalar variable handle type.")
        return float(value_arr[0]) if value_arr.size > 0 else 0.0

    @staticmethod
    def _pairs_from_readable(pairs_readable: List[_ReadablePair]) -> PairTuple:
        r"""
        Convert readable interpolation-pair records to internal tuple form.

        **Parameters**

        - `pairs_readable` (:class:`~typing.List`\[:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Union`\[:class:`int`, :class:`str`\]\]\]):
          Readable pairs of the form `{"j": ..., "k": ...}` where each entry is an
          integer or `"star"`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\], ...\]):
          Internal pair tuple used by solver-building helpers.
        """
        pairs: List[Pair] = []
        for pair in pairs_readable:
            j_val = pair["j"]
            k_val = pair["k"]
            if j_val == "star" or k_val == "star":
                pairs.append(("star", "star"))
            else:
                pairs.append((int(j_val), int(k_val)))
        return tuple(pairs)

    @staticmethod
    def _min_symmetric_eigenvalue(matrix: np.ndarray) -> float:
        r"""
        Return the smallest eigenvalue of a symmetrized matrix.

        **Parameters**

        - `matrix` (:class:`numpy.ndarray`): Matrix to symmetrize and analyze.

        **Returns**

        - (:class:`float`): Minimum eigenvalue of :math:`(matrix + matrix^\top)/2`.
        """
        symmetric_matrix = 0.5 * (matrix + matrix.T)
        try:
            eigvals = np.linalg.eigvalsh(symmetric_matrix)
            return float(np.min(eigvals))
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvals(symmetric_matrix)
            return float(np.min(np.real(eigvals)))

    @staticmethod
    def _compute_iteration_dependent_diagnostics(
            prob: InclusionProblem,
            algo: Algorithm,
            K: int,
            c_K_value: float,
            certificate: _IterationDependentCertificate,
    ) -> Dict[str, Any]:
        r"""
        Compute post-solve diagnostics for constrained scalars, PSD blocks, and equalities.

        **Parameters**

        - `prob` (:class:`~autolyap.problemclass.InclusionProblem`): Inclusion problem instance.
        - `algo` (:class:`~autolyap.algorithms.Algorithm`): Algorithm instance.
        - `K` (:class:`int`): Horizon parameter.
        - `c_K_value` (:class:`float`): Solved objective value :math:`c_K`.
        - `certificate` (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]): Extracted solver certificate.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]): Diagnostic dictionary with
          `nonnegative`, `psd`, and `equality` summaries.
        """
        Q_sequence = [np.asarray(Q_k, dtype=float) for Q_k in certificate["Q_sequence"]]
        q_sequence_raw = certificate["q_sequence"]
        q_sequence = None
        if q_sequence_raw is not None:
            q_sequence = [np.asarray(q_k, dtype=float).reshape(-1) for q_k in q_sequence_raw]
        multipliers = certificate["multipliers"]

        nonnegative_records: List[Tuple[str, float]] = [("c_K", float(c_K_value))]
        for multiplier_name in ("operator_lambda", "function_lambda"):
            for record in multipliers[multiplier_name]:
                label = (
                    f"{multiplier_name}(iteration={record['iteration']},"
                    f" component={record['component']},"
                    f" interpolation_index={record['interpolation_index']})"
                )
                nonnegative_records.append((label, float(record["value"])))

        negative_nonnegative = [(label, value) for label, value in nonnegative_records if value < 0.0]
        largest_nonnegative_violation = max((-value for _, value in negative_nonnegative), default=0.0)
        worst_nonnegative = min(nonnegative_records, key=lambda item: item[1], default=None)

        Ws: Dict[int, np.ndarray] = {}
        Theta0_0, Theta1_0 = IterationDependent._compute_Thetas(algo, 0)
        Ws[0] = Theta1_0.T @ Q_sequence[1] @ Theta1_0 - c_K_value * (Theta0_0.T @ Q_sequence[0] @ Theta0_0)
        for k in range(1, K):
            Theta0_k, Theta1_k = IterationDependent._compute_Thetas(algo, k)
            Ws[k] = Theta1_k.T @ Q_sequence[k + 1] @ Theta1_k - Theta0_k.T @ Q_sequence[k] @ Theta0_k

        psd_constraint_sums: Dict[int, np.ndarray] = {k: -np.asarray(Ws[k], dtype=float) for k in range(0, K)}

        all_psd_multiplier_records = (
            multipliers["operator_lambda"]
            + multipliers["function_lambda"]
            + multipliers["function_nu"]
        )
        component_data = {i: prob._get_component_data(i) for i in range(1, algo.m + 1)}

        for record in all_psd_multiplier_records:
            k = int(record["iteration"])
            if k not in psd_constraint_sums:
                continue

            i = int(record["component"])
            interpolation_index = int(record["interpolation_index"])
            value = float(record["value"])
            interpolation_data = component_data[i][interpolation_index]
            M = np.asarray(interpolation_data[0], dtype=float)
            if not np.any(M):
                continue

            pairs = IterationDependent._pairs_from_readable(record["pairs"])
            E_matrix = algo._compute_E(i, list(pairs), k, k + 1, validate=False)
            W_matrix = E_matrix.T @ M @ E_matrix
            psd_constraint_sums[k] = psd_constraint_sums[k] + value * W_matrix

        psd_per_constraint: List[Dict[str, Any]] = []
        largest_psd_violation = 0.0
        worst_psd_entry: Optional[Dict[str, Any]] = None

        for k in range(0, K):
            min_eigenvalue = IterationDependent._min_symmetric_eigenvalue(psd_constraint_sums[k])
            violation = max(0.0, -min_eigenvalue)
            entry = {
                "label": f"iteration={k}",
                "iteration": k,
                "min_eigenvalue": min_eigenvalue,
                "violation": violation,
            }
            psd_per_constraint.append(entry)
            if worst_psd_entry is None or min_eigenvalue < float(worst_psd_entry["min_eigenvalue"]):
                worst_psd_entry = entry
            if violation > largest_psd_violation:
                largest_psd_violation = violation

        equality_per_constraint: List[Dict[str, Any]] = []
        violating_equality_entries = 0
        total_equality_entries = 0
        largest_equality_violation = 0.0
        worst_equality_entry: Optional[Dict[str, Any]] = None
        equality_tolerance = 1e-9

        if algo.m_func > 0:
            if q_sequence is None:
                raise ValueError("Certificate is missing q_sequence while functional components are active.")
            m_bar_func = algo.m_bar_func
            m_func = algo.m_func
            theta0, theta1 = IterationDependent._compute_thetas(algo)

            ws: Dict[int, np.ndarray] = {}
            ws[0] = theta1.T @ q_sequence[1] - c_K_value * (theta0.T @ q_sequence[0])
            for k in range(1, K):
                ws[k] = theta1.T @ q_sequence[k + 1] - theta0.T @ q_sequence[k]

            eq_constraint_sums: Dict[int, np.ndarray] = {
                k: -np.asarray(ws[k], dtype=float).reshape(-1) for k in range(0, K)
            }

            eq_multiplier_records = multipliers["function_lambda"] + multipliers["function_nu"]
            lifted_F_basis_cache: Dict[Tuple[int, int, PairTuple], np.ndarray] = {}

            def _get_lifted_F_basis(k_idx: int, i: int, pairs: PairTuple) -> np.ndarray:
                r"""
                Return cached lifted F basis for one iteration/component/pair pattern.

                **Parameters**

                - `k_idx` (:class:`int`): Iteration index.
                - `i` (:class:`int`): Component index.
                - `pairs` (:class:`PairTuple`): Pair pattern to lift.

                **Returns**

                - (:class:`numpy.ndarray`): Lifted F basis matrix.
                """
                cache_key = (k_idx, i, pairs)
                F_basis = lifted_F_basis_cache.get(cache_key)
                if F_basis is None:
                    Fs_dict = algo._get_Fs(k_idx, k_idx + 1)
                    total_dim = 2 * m_bar_func + m_func
                    F_basis = np.empty((total_dim, len(pairs)))
                    for col_idx, (j, k_pair) in enumerate(pairs):
                        key = (i, "star", "star") if (j == "star" and k_pair == "star") else (i, j, k_pair)
                        F_basis[:, col_idx] = Fs_dict[key].reshape(-1)
                    lifted_F_basis_cache[cache_key] = F_basis
                return F_basis

            component_data = {i: prob._get_component_data(i) for i in range(1, algo.m + 1)}
            for record in eq_multiplier_records:
                k = int(record["iteration"])
                if k not in eq_constraint_sums:
                    continue

                i = int(record["component"])
                interpolation_index = int(record["interpolation_index"])
                value = float(record["value"])
                interpolation_data = component_data[i][interpolation_index]
                a_vec = np.asarray(interpolation_data[1], dtype=float).reshape(-1)
                if not np.any(a_vec):
                    continue

                pairs = IterationDependent._pairs_from_readable(record["pairs"])
                F_basis = _get_lifted_F_basis(k, i, pairs)
                F_vector = (F_basis @ a_vec).reshape(-1)
                eq_constraint_sums[k] = eq_constraint_sums[k] + value * F_vector

            for k in range(0, K):
                residual = np.asarray(eq_constraint_sums[k], dtype=float).reshape(-1)
                max_abs = float(np.max(np.abs(residual))) if residual.size > 0 else 0.0
                l2_norm = float(np.linalg.norm(residual))
                argmax_index = int(np.argmax(np.abs(residual))) if residual.size > 0 else -1
                signed_value = float(residual[argmax_index]) if argmax_index >= 0 else 0.0
                violating_entries = int(np.sum(np.abs(residual) > equality_tolerance))
                violating_equality_entries += violating_entries
                total_equality_entries += int(residual.size)

                entry = {
                    "label": f"iteration={k}",
                    "iteration": k,
                    "dimension": int(residual.size),
                    "max_abs_residual": max_abs,
                    "l2_residual": l2_norm,
                    "argmax_index": argmax_index,
                    "signed_residual_at_argmax": signed_value,
                }
                equality_per_constraint.append(entry)

                if max_abs > largest_equality_violation:
                    largest_equality_violation = max_abs
                    worst_equality_entry = entry

        return {
            "nonnegative": {
                "total_count": len(nonnegative_records),
                "negative_count": len(negative_nonnegative),
                "largest_violation": largest_nonnegative_violation,
                "worst_label": None if worst_nonnegative is None else worst_nonnegative[0],
                "worst_value": None if worst_nonnegative is None else worst_nonnegative[1],
                "top_negative": sorted(negative_nonnegative, key=lambda item: item[1])[:5],
            },
            "psd": {
                "total_count": len(psd_per_constraint),
                "negative_count": sum(1 for entry in psd_per_constraint if float(entry["violation"]) > 0.0),
                "largest_violation": largest_psd_violation,
                "worst_label": None if worst_psd_entry is None else worst_psd_entry["label"],
                "worst_min_eigenvalue": None if worst_psd_entry is None else worst_psd_entry["min_eigenvalue"],
                "per_constraint": psd_per_constraint,
            },
            "equality": {
                "total_count": len(equality_per_constraint),
                "total_entries": total_equality_entries,
                "violating_entries": violating_equality_entries,
                "tolerance": equality_tolerance,
                "largest_violation": largest_equality_violation,
                "worst_label": None if worst_equality_entry is None else worst_equality_entry["label"],
                "worst_index": None if worst_equality_entry is None else worst_equality_entry["argmax_index"],
                "worst_signed_residual": (
                    None if worst_equality_entry is None else worst_equality_entry["signed_residual_at_argmax"]
                ),
                "per_constraint": equality_per_constraint,
            },
        }

    @staticmethod
    def _print_iteration_dependent_diagnostics(
            diagnostics: Dict[str, Any],
            K: int,
            c_K_value: float,
            backend: str,
            verbosity: int,
    ) -> None:
        r"""
        Print user-facing diagnostics for iteration-dependent verification.

        **Parameters**

        - `diagnostics` (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]): Diagnostic payload from
          :meth:`~autolyap.iteration_dependent.IterationDependent._compute_iteration_dependent_diagnostics`.
        - `K` (:class:`int`): Horizon parameter.
        - `c_K_value` (:class:`float`): Solved objective value :math:`c_K`.
        - `backend` (:class:`str`): Solver backend label.
        - `verbosity` (:class:`int`): Output level (`0`, `1`, or `2+`).

        **Returns**

        - `None`: Prints diagnostics to stdout.
        """
        if verbosity <= 0:
            return

        nonnegative = diagnostics["nonnegative"]
        psd = diagnostics["psd"]
        equality = diagnostics["equality"]

        print(
            f"[AutoLyap][INFO] Iteration-dependent SDP diagnostics "
            f"(backend={backend}, K={K}, c_K={c_K_value:.12g})."
        )
        print(
            f"[AutoLyap][INFO] Nonnegativity check: "
            f"{nonnegative['negative_count']}/{nonnegative['total_count']} constrained scalars are negative; "
            f"largest violation={nonnegative['largest_violation']:.3e}."
        )
        if nonnegative["worst_label"] is not None:
            if int(nonnegative["negative_count"]) > 0:
                print(
                    f"[AutoLyap][INFO] Worst violating constrained scalar value: "
                    f"{float(nonnegative['worst_value']):.3e} at {nonnegative['worst_label']}."
                )
            else:
                print(
                    f"[AutoLyap][INFO] Smallest constrained scalar value (nonnegative): "
                    f"{float(nonnegative['worst_value']):.3e} at {nonnegative['worst_label']}."
                )

        print(
            f"[AutoLyap][INFO] PSD check: "
            f"{psd['negative_count']}/{psd['total_count']} constrained matrices have a negative minimum eigenvalue; "
            f"largest violation={psd['largest_violation']:.3e}."
        )
        if psd["worst_label"] is not None:
            if int(psd["negative_count"]) > 0:
                print(
                    f"[AutoLyap][INFO] Worst PSD minimum eigenvalue: "
                    f"{float(psd['worst_min_eigenvalue']):.3e} at {psd['worst_label']}."
                )
            else:
                print(
                    f"[AutoLyap][INFO] Smallest PSD minimum eigenvalue (nonnegative): "
                    f"{float(psd['worst_min_eigenvalue']):.3e} at {psd['worst_label']}."
                )

        if equality["total_count"] == 0:
            print("[AutoLyap][INFO] Equality check: no active equality constraints.")
        else:
            print(
                f"[AutoLyap][INFO] Equality check: "
                f"{equality['violating_entries']}/{equality['total_entries']} entries exceed "
                f"tol={float(equality['tolerance']):.1e}; "
                f"largest absolute residual={float(equality['largest_violation']):.3e}."
            )
            if equality["worst_label"] is not None:
                print(
                    f"[AutoLyap][INFO] Worst equality residual: "
                    f"{float(equality['worst_signed_residual']):.3e} at "
                    f"{equality['worst_label']}[index={int(equality['worst_index'])}]."
                )

        if verbosity >= 2:
            for entry in psd["per_constraint"]:
                print(
                    f"[AutoLyap][DETAIL] {entry['label']}: "
                    f"min_eigenvalue={float(entry['min_eigenvalue']):.3e}, "
                    f"violation={float(entry['violation']):.3e}."
                )
            for entry in equality["per_constraint"]:
                print(
                    f"[AutoLyap][DETAIL] {entry['label']}: "
                    f"max_abs_residual={float(entry['max_abs_residual']):.3e}, "
                    f"l2_residual={float(entry['l2_residual']):.3e}."
                )
            for label, value in nonnegative["top_negative"]:
                print(
                    f"[AutoLyap][DETAIL] Negative constrained scalar: "
                    f"value={value:.3e} at {label}."
                )

    @staticmethod
    def _validate_iteration_dependent_inputs(
            prob: InclusionProblem,
            algo: Algorithm,
            K: int,
            Q_0: np.ndarray,
            Q_K: np.ndarray,
            q_0: Optional[np.ndarray],
            q_K: Optional[np.ndarray],
        ) -> Tuple[int, int, int, int, int, int, int, int, int, Optional[int]]:
        r"""
        Validate problem/algo consistency and endpoint Lyapunov parameter shapes.

        **Parameters**

        - `prob` (:class:`~autolyap.problemclass.InclusionProblem`): Inclusion problem instance.
        - `algo` (:class:`~autolyap.algorithms.Algorithm`): Algorithm instance.
        - `K` (:class:`int`): Candidate horizon.
        - `Q_0` (:class:`numpy.ndarray`): Endpoint Lyapunov matrix at iteration `0`.
        - `Q_K` (:class:`numpy.ndarray`): Endpoint Lyapunov matrix at iteration `K`.
        - `q_0` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): Endpoint linear term at iteration `0`.
        - `q_K` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): Endpoint linear term at iteration `K`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`int`, :class:`~typing.Optional`\[:class:`int`\]\]):
          Normalized tuple `(K, n, m_bar, m, m_bar_func, m_func, m_op, m_bar_op, dim_Q, dim_q)`.

        **Raises**

        - `ValueError`: If dimensions, symmetry, finiteness, or component-index compatibility checks fail.
        """
        # Validate consistency between the problem and the algorithm.
        if prob.m != algo.m:
            raise ValueError("Mismatch in number of components: prob.m and algo.m must be the same")

        # Check that functional/operator component indices are identical.
        if set(prob.I_func) != set(algo.I_func):
            raise ValueError("Mismatch in functional component indices between prob and algo")
        if set(prob.I_op) != set(algo.I_op):
            raise ValueError("Mismatch in operator component indices between prob and algo")

        # Ensure K is positive.
        K = ensure_integral(K, "K", minimum=1)

        # Retrieve dimensions from the algorithm instance.
        n = algo.n                      # State dimension.
        m_bar = algo.m_bar              # Total evaluations per iteration.
        m = algo.m                      # Total number of components.
        m_bar_func = algo.m_bar_func    # Total evaluations for functional components.
        m_func = algo.m_func            # Number of functional components.
        m_op = algo.m_op                # Number of operator components.
        m_bar_op = algo.m_bar_op        # Total evaluations for operator components.

        # Expected dimension for Q matrices: [n + m_bar + m] x [n + m_bar + m].
        dim_Q = n + m_bar + m
        if not (isinstance(Q_0, np.ndarray) and Q_0.ndim == 2 and Q_0.shape[0] == Q_0.shape[1] == dim_Q):
            raise ValueError(
                f"Q_0 must be a symmetric matrix of dimension {dim_Q}x{dim_Q}. "
                f"Got shape {getattr(Q_0, 'shape', None)}."
            )
        ensure_finite_array(Q_0, "Q_0")
        if not np.allclose(Q_0, Q_0.T, atol=1e-8):
            raise ValueError("Q_0 must be symmetric.")
        if not (isinstance(Q_K, np.ndarray) and Q_K.ndim == 2 and Q_K.shape[0] == Q_K.shape[1] == dim_Q):
            raise ValueError(
                f"Q_K must be a symmetric matrix of dimension {dim_Q}x{dim_Q}. "
                f"Got shape {getattr(Q_K, 'shape', None)}."
            )
        ensure_finite_array(Q_K, "Q_K")
        if not np.allclose(Q_K, Q_K.T, atol=1e-8):
            raise ValueError("Q_K must be symmetric.")

        # For functional components, q_0 and q_K must have proper dimensions.
        dim_q = None
        if m_func > 0:
            dim_q = m_bar_func + m_func

            # Check q_0
            if q_0 is None:
                raise ValueError(f"q_0 must be a 1D numpy array of length {dim_q}, but got None.")
            if not (isinstance(q_0, np.ndarray) and q_0.ndim == 1 and q_0.shape[0] == dim_q):
                raise ValueError(
                    f"q_0 must be a 1D numpy array of length {dim_q}. Got shape {getattr(q_0, 'shape', None)}."
                )
            ensure_finite_array(q_0, "q_0")

            # Check q_K
            if q_K is None:
                raise ValueError(f"q_K must be a 1D numpy array of length {dim_q}, but got None.")
            if not (isinstance(q_K, np.ndarray) and q_K.ndim == 1 and q_K.shape[0] == dim_q):
                raise ValueError(
                    f"q_K must be a 1D numpy array of length {dim_q}. Got shape {getattr(q_K, 'shape', None)}."
                )
            ensure_finite_array(q_K, "q_K")
        else:
            if q_0 is not None or q_K is not None:
                raise ValueError("q_0 and q_K must be None when there are no functional components.")

        return K, n, m_bar, m, m_bar_func, m_func, m_op, m_bar_op, dim_Q, dim_q

    @staticmethod
    def _expected_pairs_len(interp_key: str) -> int:
        r"""
        Return the expected interpolation-pair arity for one index-pattern key.

        **Parameters**

        - `interp_key` (:class:`str`): Interpolation index key such as `"r1"` or `"r1<r2"`.

        **Returns**

        - (:class:`int`): Expected number of pairs for that key.

        **Raises**

        - `ValueError`: If `interp_key` is unknown.
        """
        if interp_key == 'r1':
            return 1
        if interp_key in ('r1<r2', 'r1!=r2', 'r1!=star'):
            return 2
        raise ValueError(f"Error: Invalid interpolation indices: {interp_key}.")

    @staticmethod
    def _iter_pair_patterns(
            interp_key: str,
            pairs_with_star: List[Pair],
            pairs_no_star: List[Tuple[int, int]],
            star_pair: Pair,
    ) -> Iterator[PairTuple]:
        r"""
        Yield concrete pair tuples that satisfy one interpolation-index pattern.

        **Parameters**

        - `interp_key` (:class:`str`): Pattern key (`"r1"`, `"r1<r2"`, `"r1!=r2"`, `"r1!=star"`).
        - `pairs_with_star` (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\]\]):
          Candidate non-star pairs plus the star pair.
        - `pairs_no_star` (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\]\]): Candidate non-star pairs only.
        - `star_pair` (:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\]): Star pair token.

        **Yields**

        - (:class:`~typing.Tuple`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\], ...\]):
          Pair tuples compatible with `interp_key`.

        **Raises**

        - `ValueError`: If `interp_key` is unknown.
        """
        if interp_key == 'r1':
            for pair in pairs_with_star:
                yield (pair,)
            return

        if interp_key == 'r1<r2':
            yield from combinations(pairs_with_star, 2)
            return

        if interp_key == 'r1!=r2':
            n_pairs = len(pairs_with_star)
            for idx1 in range(n_pairs):
                pair1 = pairs_with_star[idx1]
                for idx2 in range(n_pairs):
                    if idx1 == idx2:
                        continue
                    yield (pair1, pairs_with_star[idx2])
            return

        if interp_key == 'r1!=star':
            for pair in pairs_no_star:
                yield (pair, star_pair)
            return

        raise ValueError(f"Error: Invalid interpolation indices: {interp_key}.")

    @staticmethod
    def _collect_iteration_dependent_component_data(
            prob: InclusionProblem,
            m: int,
            op_components: set,
    ) -> Dict[int, List[Tuple[InterpolationData, str, bool, bool]]]:
        r"""
        Validate interpolation payloads and cache per-component metadata.

        **Parameters**

        - `prob` (:class:`~autolyap.problemclass.InclusionProblem`): Inclusion problem instance.
        - `m` (:class:`int`): Number of components.
        - `op_components` (:class:`set`): Set of operator-component indices.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`int`, :class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~typing.Any`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~typing.Any`\]\], :class:`str`, :class:`bool`, :class:`bool`\]\]\]):
          Per-component validated metadata used by SDP builders.

        **Raises**

        - `ValueError`: If interpolation matrix/vector shapes are inconsistent with interpolation indices.
        """
        component_data: Dict[int, List[Tuple[InterpolationData, str, bool, bool]]] = {}
        for i in range(1, m + 1):
            is_op = i in op_components
            data = prob._get_component_data(i)
            validated: List[Tuple[InterpolationData, str, bool, bool]] = []
            for o, interp_data in enumerate(data):
                if is_op:
                    interp_idx = cast(OperatorInterpolationData, interp_data)[1]
                else:
                    interp_idx = cast(FunctionInterpolationData, interp_data)[3]
                interp_key = str(interp_idx)
                expected_len = IterationDependent._expected_pairs_len(interp_key)
                expected_dim = 2 * expected_len

                if is_op:
                    interp_data_op = cast(OperatorInterpolationData, interp_data)
                    M, _ = interp_data_op
                    if getattr(M, 'shape', None) != (expected_dim, expected_dim):
                        raise ValueError(
                            f"Interpolation matrix for component {i}, condition {o} must have "
                            f"shape ({expected_dim}, {expected_dim}) for indices {interp_key}. "
                            f"Got {getattr(M, 'shape', None)}."
                        )
                    has_quadratic = bool(np.any(M))
                    validated.append((interp_data_op, interp_key, has_quadratic, False))
                else:
                    interp_data_func = cast(FunctionInterpolationData, interp_data)
                    M, a, _eq, _ = interp_data_func
                    if getattr(a, 'shape', None) != (expected_len,):
                        raise ValueError(
                            f"Interpolation vector for component {i}, condition {o} must have "
                            f"length {expected_len} for indices {interp_key}. Got {getattr(a, 'shape', None)}."
                        )
                    if getattr(M, 'shape', None) != (expected_dim, expected_dim):
                        raise ValueError(
                            f"Interpolation matrix for component {i}, condition {o} must have "
                            f"shape ({expected_dim}, {expected_dim}) for indices {interp_key}. "
                            f"Got {getattr(M, 'shape', None)}."
                        )
                    has_quadratic = bool(np.any(M))
                    has_linear = bool(np.any(a))
                    validated.append((interp_data_func, interp_key, has_quadratic, has_linear))
            component_data[i] = validated
        return component_data

    @staticmethod
    def _build_iteration_dependent_model(
            prob: InclusionProblem,
            algo: Algorithm,
            K: int,
            Q_0: np.ndarray,
            Q_K: np.ndarray,
            q_0: Optional[np.ndarray],
            q_K: Optional[np.ndarray],
            dim_Q: int,
            dim_q: Optional[int],
            m_func: int,
            m_op: int,
            model: Optional[Any] = None,
    ) -> Tuple[Any, _IterationDependentMosekSolutionHandles]:
        r"""
        Assemble the iteration-dependent MOSEK Fusion model.

        This routine creates optimization variables, accumulates PSD/equality
        constraints from interpolation conditions, and returns solver handles used
        for certificate extraction.

        **Parameters**

        - `prob` (:class:`~autolyap.problemclass.InclusionProblem`): Inclusion problem instance.
        - `algo` (:class:`~autolyap.algorithms.Algorithm`): Algorithm instance.
        - `K` (:class:`int`): Horizon parameter.
        - `Q_0`, `Q_K`: Endpoint Lyapunov matrices.
        - `q_0`, `q_K`: Endpoint linear terms (or `None` when no functional components).
        - `dim_Q` (:class:`int`): Dimension of each `Q_k`.
        - `dim_q` (:class:`~typing.Optional`\[:class:`int`\]): Dimension of each `q_k`.
        - `m_func` (:class:`int`): Number of functional components.
        - `m_op` (:class:`int`): Number of operator components.
        - `model` (:class:`~typing.Optional`\[:class:`mosek.fusion.Model`\]): Optional existing model.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`mosek.fusion.Model`, :class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]\]):
          The built model and extraction handles.
        """
        mf = IterationDependent._import_mosek_fusion()
        Mod = model if model is not None else mf.Model()
        c_K = Mod.variable("c_K", 1, mf.Domain.greaterThan(0.0))

        Qs = {}
        Qs[0] = Q_0
        Qs[K] = Q_K
        Qij_vars: Dict[int, MosekUpperTriangleSolutionHandleProtocol] = {}
        for k in range(1, K):
            Qij = Mod.variable(f"Q_{k}", dim_Q * (dim_Q + 1) // 2, mf.Domain.unbounded())
            Q_k = create_symmetric_matrix_expression(Qij, dim_Q)
            Qs[k] = Q_k
            Qij_vars[k] = Qij

        q_vars: Dict[int, MosekLevelHandleProtocol] = {}
        if m_func > 0:
            qs = {}
            qs[0] = q_0
            qs[K] = q_K
            for k in range(1, K):
                q_k = Mod.variable(f"q_{k}", dim_q, mf.Domain.unbounded())
                qs[k] = q_k
                q_vars[k] = q_k

        # Build the main PSD and equality-constraint sums.
        Ws = {}
        (Theta0, Theta1) = IterationDependent._compute_Thetas(algo, 0)
        # First inequality uses the scaled Lyapunov decrease with multiplier c_K.
        W_0 = Theta1.T @ Qs[1] @ Theta1 - c_K[0] * Theta0.T @ Qs[0] @ Theta0
        Ws[0] = W_0
        for k in range(1, K):
            (Theta0, Theta1) = IterationDependent._compute_Thetas(algo, k)
            W_k = Theta1.T @ Qs[k+1] @ Theta1 - Theta0.T @ Qs[k] @ Theta0
            Ws[k] = W_k

        if m_func > 0:
            ws = {}
            (theta0, theta1) = IterationDependent._compute_thetas(algo)
            # Linear term for functional components mirrors the quadratic Lyapunov recursion.
            w_0 = theta1.T @ qs[1] - c_K[0] * theta0.T @ qs[0]
            ws[0] = w_0
            for k in range(1, K):
                w_k = theta1.T @ qs[k+1] - theta0.T @ qs[k]
                ws[k] = w_k

        # Initialize dictionaries to accumulate PSD and equality constraints.
        PSD_constraint_sums = {}
        eq_constraint_sums = {}
        for k in range(0, K):
            PSD_constraint_sums[k] = -Ws[k]
            if m_func > 0:
                eq_constraint_sums[k] = -ws[k]

        # Multipliers for interpolation conditions.
        lambdas_op: IterationDependentMultiplierMap = {}
        lambdas_func: IterationDependentMultiplierMap = {}
        nus_func: IterationDependentMultiplierMap = {}

        # Inner helpers for processing interpolation data.
        n = algo.n
        m_bar = algo.m_bar
        m_bar_func = algo.m_bar_func
        op_components = set(algo.I_op)
        m = algo.m
        m_bar_is = algo.m_bar_is
        _compute_E = algo._compute_E
        _get_Fs = algo._get_Fs
        mod_variable = Mod.variable
        domain_ge0 = mf.Domain.greaterThan(0.0)
        domain_unbounded = mf.Domain.unbounded()
        star_pair = ('star', 'star')
        lifted_E_cache: Dict[Tuple[int, int, Tuple[Union[Tuple[int, int], Tuple[str, str]], ...]], np.ndarray] = {}
        lifted_F_basis_cache: Dict[Tuple[int, int, Tuple[Union[Tuple[int, int], Tuple[str, str]], ...]], np.ndarray] = {}

        def _get_lifted_E(k: int, i: int, pairs: PairTuple) -> np.ndarray:
            r"""
            Return cached lifted E matrix for one iteration/component/pair pattern.

            **Parameters**

            - `k` (:class:`int`): Iteration index.
            - `i` (:class:`int`): Component index.
            - `pairs` (:class:`PairTuple`): Pair pattern to lift.

            **Returns**

            - (:class:`numpy.ndarray`): Lifted E matrix.
            """
            cache_key = (k, i, pairs)
            E_matrix = lifted_E_cache.get(cache_key)
            if E_matrix is None:
                E_matrix = _compute_E(i, list(pairs), k, k + 1, validate=False)
                lifted_E_cache[cache_key] = E_matrix
            return E_matrix

        def _get_lifted_F_basis(k: int, i: int, pairs: PairTuple) -> np.ndarray:
            r"""
            Return cached lifted F basis for one iteration/component/pair pattern.

            **Parameters**

            - `k` (:class:`int`): Iteration index.
            - `i` (:class:`int`): Component index.
            - `pairs` (:class:`PairTuple`): Pair pattern to lift.

            **Returns**

            - (:class:`numpy.ndarray`): Lifted F basis matrix.
            """
            cache_key = (k, i, pairs)
            F_basis = lifted_F_basis_cache.get(cache_key)
            if F_basis is None:
                Fs_dict = _get_Fs(k, k + 1)
                total_dim = 2 * m_bar_func + m_func
                F_basis = np.empty((total_dim, len(pairs)))
                for col_idx, (j, k_idx) in enumerate(pairs):
                    key = (i, 'star', 'star') if (j == 'star' and k_idx == 'star') else (i, j, k_idx)
                    F_basis[:, col_idx] = Fs_dict[key].reshape(-1)
                lifted_F_basis_cache[cache_key] = F_basis
            return F_basis

        def process_pairs(k: int,
                          i: int,
                          o: int,
                          interpolation_data: InterpolationData,
                          pairs: PairTuple,
                          comp_type: str,
                          has_quadratic: bool,
                          has_linear: bool) -> None:
            r"""
            Internal helper for one interpolation-pair pattern.

            Creates multiplier variables and accumulates contributions to the
            PSD/equality constraints at iteration :math:`k`.

            **Parameters**

            - `k` (:class:`int`): Iteration index.
            - `i` (:class:`int`): Component index.
            - `o` (:class:`int`): Interpolation condition index.
            - `interpolation_data` (:class:`InterpolationData`): Interpolation payload.
            - `pairs` (:class:`PairTuple`): Concrete pair pattern.
            - `comp_type` (:class:`str`): `"op"` or `"func"`.
            - `has_quadratic` (:class:`bool`): Whether quadratic terms are present.
            - `has_linear` (:class:`bool`): Whether linear terms are present.

            **Returns**

            - `None`: Accumulates constraint contributions in place.
            """
            key = (k, i, pairs, o)

            if comp_type == 'op':
                if not has_quadratic:
                    return
                M, _ = cast(OperatorInterpolationData, interpolation_data)
            else:
                M, a, eq, _ = cast(FunctionInterpolationData, interpolation_data)
                if not has_quadratic and not has_linear:
                    return

            W_matrix = None
            if has_quadratic:
                E_matrix = _get_lifted_E(k, i, pairs)
                W_matrix = E_matrix.T @ M @ E_matrix

            if comp_type == 'op':
                lambda_var = mod_variable(1, domain_ge0)
                lambdas_op[key] = lambda_var
                PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var[0] * W_matrix
            else:
                F_vector = None
                if has_linear:
                    F_basis = _get_lifted_F_basis(k, i, pairs)
                    F_vector = (F_basis @ a).reshape(-1, 1)
                if eq:
                    nu_var = mod_variable(1, domain_unbounded)
                    nus_func[key] = nu_var
                    if has_quadratic:
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + nu_var[0] * W_matrix
                    if has_linear:
                        eq_constraint_sums[k] = eq_constraint_sums[k] + nu_var[0] * F_vector
                else:
                    lambda_var = mod_variable(1, domain_ge0)
                    lambdas_func[key] = lambda_var
                    if has_quadratic:
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var[0] * W_matrix
                    if has_linear:
                        eq_constraint_sums[k] = eq_constraint_sums[k] + lambda_var[0] * F_vector
        component_data = IterationDependent._collect_iteration_dependent_component_data(
            prob, m, op_components
        )

        # Loop over iterations and process interpolation constraints via explicit pair-pattern iterators.
        for k in range(0, K):
            k_range = range(k, k + 2)
            for i in range(1, m + 1):
                m_bar_i = m_bar_is[i - 1]
                pairs_no_star = [(j, k_) for j in range(1, m_bar_i + 1) for k_ in k_range]
                pairs_with_star = pairs_no_star + [star_pair]
                is_op = i in op_components
                comp_type = 'op' if is_op else 'func'
                for o, (interp_data, interp_key, has_quadratic, has_linear) in enumerate(component_data[i]):
                    for pair_pattern in IterationDependent._iter_pair_patterns(
                        interp_key, pairs_with_star, pairs_no_star, star_pair
                    ):
                        process_pairs(
                            k,
                            i,
                            o,
                            interp_data,
                            pair_pattern,
                            comp_type,
                            has_quadratic,
                            has_linear,
                        )

        for k in range(0, K):
            Mod.constraint(PSD_constraint_sums[k], mf.Domain.inPSDCone(n + 2 * m_bar + m))
            if m_func > 0:
                Mod.constraint(eq_constraint_sums[k] == 0)

        Mod.objective("obj", mf.ObjectiveSense.Minimize, c_K)

        solution_handles: _IterationDependentMosekSolutionHandles = {
            "K": K,
            "dim_Q": dim_Q,
            "dim_q": dim_q,
            "m_func": m_func,
            "Q_0": Q_0,
            "Q_K": Q_K,
            "q_0": q_0,
            "q_K": q_K,
            "Qij_vars": Qij_vars,
            "q_vars": q_vars,
            "lambdas_op": lambdas_op,
            "lambdas_func": lambdas_func,
            "nus_func": nus_func,
            "c_K_var": c_K,
        }
        return Mod, solution_handles

    @staticmethod
    def _build_iteration_dependent_problem_cvxpy(
            prob: InclusionProblem,
            algo: Algorithm,
            K: int,
            Q_0: np.ndarray,
            Q_K: np.ndarray,
            q_0: Optional[np.ndarray],
            q_K: Optional[np.ndarray],
            dim_Q: int,
            dim_q: Optional[int],
            m_func: int,
            m_op: int,
            cp: CvxpyModuleProtocol,
    ) -> Tuple[Any, _IterationDependentCvxpySolutionHandles]:
        r"""
        Assemble the CVXPY iteration-dependent problem and extraction handles.

        **Parameters**

        - `prob` (:class:`~autolyap.problemclass.InclusionProblem`): Inclusion problem instance.
        - `algo` (:class:`~autolyap.algorithms.Algorithm`): Algorithm instance.
        - `K` (:class:`int`): Horizon parameter.
        - `Q_0`, `Q_K`: Endpoint Lyapunov matrices.
        - `q_0`, `q_K`: Endpoint linear terms (or `None` when no functional components).
        - `dim_Q` (:class:`int`): Dimension of each `Q_k`.
        - `dim_q` (:class:`~typing.Optional`\[:class:`int`\]): Dimension of each `q_k`.
        - `m_func` (:class:`int`): Number of functional components.
        - `m_op` (:class:`int`): Number of operator components.
        - `cp`: Imported CVXPY module.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`~typing.Any`, :class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]\]):
          The CVXPY problem and extraction handles.
        """
        c_K = cp.Variable(nonneg=True)

        Qs: Dict[int, Any] = {0: Q_0, K: Q_K}
        Q_vars: Dict[int, CvxpyValueHandleProtocol] = {}
        for k in range(1, K):
            Q_k = cp.Variable((dim_Q, dim_Q), symmetric=True)
            Qs[k] = Q_k
            Q_vars[k] = Q_k

        q_vars: Dict[int, CvxpyValueHandleProtocol] = {}
        if m_func > 0:
            qs: Dict[int, Any] = {0: q_0, K: q_K}
            for k in range(1, K):
                q_k = cp.Variable(dim_q)
                qs[k] = q_k
                q_vars[k] = q_k

        Ws: Dict[int, Any] = {}
        Theta0, Theta1 = IterationDependent._compute_Thetas(algo, 0)
        Ws[0] = Theta1.T @ Qs[1] @ Theta1 - c_K * (Theta0.T @ Qs[0] @ Theta0)
        for k in range(1, K):
            Theta0, Theta1 = IterationDependent._compute_Thetas(algo, k)
            Ws[k] = Theta1.T @ Qs[k + 1] @ Theta1 - Theta0.T @ Qs[k] @ Theta0

        if m_func > 0:
            ws: Dict[int, Any] = {}
            theta0, theta1 = IterationDependent._compute_thetas(algo)
            ws[0] = theta1.T @ qs[1] - c_K * (theta0.T @ qs[0])
            for k in range(1, K):
                ws[k] = theta1.T @ qs[k + 1] - theta0.T @ qs[k]

        PSD_constraint_sums: Dict[int, Any] = {}
        eq_constraint_sums: Dict[int, Any] = {}
        for k in range(0, K):
            PSD_constraint_sums[k] = -Ws[k]
            if m_func > 0:
                eq_constraint_sums[k] = -ws[k]

        lambdas_op: IterationDependentMultiplierMap = {}
        lambdas_func: IterationDependentMultiplierMap = {}
        nus_func: IterationDependentMultiplierMap = {}

        m_bar_func = algo.m_bar_func
        op_components = set(algo.I_op)
        m = algo.m
        m_bar_is = algo.m_bar_is
        _compute_E = algo._compute_E
        _get_Fs = algo._get_Fs
        star_pair = ('star', 'star')
        lifted_E_cache: Dict[Tuple[int, int, Tuple[Union[Tuple[int, int], Tuple[str, str]], ...]], np.ndarray] = {}
        lifted_F_basis_cache: Dict[Tuple[int, int, Tuple[Union[Tuple[int, int], Tuple[str, str]], ...]], np.ndarray] = {}

        def _get_lifted_E(k: int, i: int, pairs: PairTuple) -> np.ndarray:
            r"""
            Return cached lifted E matrix for one iteration/component/pair pattern.

            **Parameters**

            - `k` (:class:`int`): Iteration index.
            - `i` (:class:`int`): Component index.
            - `pairs` (:class:`PairTuple`): Pair pattern to lift.

            **Returns**

            - (:class:`numpy.ndarray`): Lifted E matrix.
            """
            cache_key = (k, i, pairs)
            E_matrix = lifted_E_cache.get(cache_key)
            if E_matrix is None:
                E_matrix = _compute_E(i, list(pairs), k, k + 1, validate=False)
                lifted_E_cache[cache_key] = E_matrix
            return E_matrix

        def _get_lifted_F_basis(k: int, i: int, pairs: PairTuple) -> np.ndarray:
            r"""
            Return cached lifted F basis for one iteration/component/pair pattern.

            **Parameters**

            - `k` (:class:`int`): Iteration index.
            - `i` (:class:`int`): Component index.
            - `pairs` (:class:`PairTuple`): Pair pattern to lift.

            **Returns**

            - (:class:`numpy.ndarray`): Lifted F basis matrix.
            """
            cache_key = (k, i, pairs)
            F_basis = lifted_F_basis_cache.get(cache_key)
            if F_basis is None:
                Fs_dict = _get_Fs(k, k + 1)
                total_dim = 2 * m_bar_func + m_func
                F_basis = np.empty((total_dim, len(pairs)))
                for col_idx, (j, k_idx) in enumerate(pairs):
                    key = (i, 'star', 'star') if (j == 'star' and k_idx == 'star') else (i, j, k_idx)
                    F_basis[:, col_idx] = Fs_dict[key].reshape(-1)
                lifted_F_basis_cache[cache_key] = F_basis
            return F_basis

        def process_pairs(
                k: int,
                i: int,
                o: int,
                interpolation_data: InterpolationData,
                pairs: PairTuple,
                comp_type: str,
                has_quadratic: bool,
                has_linear: bool) -> None:
            r"""Accumulate interpolation contributions for one CVXPY pair pattern."""
            key = (k, i, pairs, o)

            if comp_type == 'op':
                if not has_quadratic:
                    return
                M, _ = cast(OperatorInterpolationData, interpolation_data)
            else:
                M, a, eq, _ = cast(FunctionInterpolationData, interpolation_data)
                if not has_quadratic and not has_linear:
                    return

            W_matrix = None
            if has_quadratic:
                E_matrix = _get_lifted_E(k, i, pairs)
                W_matrix = E_matrix.T @ M @ E_matrix

            if comp_type == 'op':
                lambda_var = cp.Variable(nonneg=True)
                lambdas_op[key] = lambda_var
                PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var * W_matrix
            else:
                F_vector = None
                if has_linear:
                    F_basis = _get_lifted_F_basis(k, i, pairs)
                    F_vector = (F_basis @ a).reshape(-1)
                if eq:
                    nu_var = cp.Variable()
                    nus_func[key] = nu_var
                    if has_quadratic:
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + nu_var * W_matrix
                    if has_linear:
                        eq_constraint_sums[k] = eq_constraint_sums[k] + nu_var * F_vector
                else:
                    lambda_var = cp.Variable(nonneg=True)
                    lambdas_func[key] = lambda_var
                    if has_quadratic:
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var * W_matrix
                    if has_linear:
                        eq_constraint_sums[k] = eq_constraint_sums[k] + lambda_var * F_vector

        component_data = IterationDependent._collect_iteration_dependent_component_data(
            prob, m, op_components
        )

        for k in range(0, K):
            k_range = range(k, k + 2)
            for i in range(1, m + 1):
                m_bar_i = m_bar_is[i - 1]
                pairs_no_star = [(j, k_) for j in range(1, m_bar_i + 1) for k_ in k_range]
                pairs_with_star = pairs_no_star + [star_pair]
                is_op = i in op_components
                comp_type = 'op' if is_op else 'func'
                for o, (interp_data, interp_key, has_quadratic, has_linear) in enumerate(component_data[i]):
                    for pair_pattern in IterationDependent._iter_pair_patterns(
                        interp_key, pairs_with_star, pairs_no_star, star_pair
                    ):
                        process_pairs(
                            k,
                            i,
                            o,
                            interp_data,
                            pair_pattern,
                            comp_type,
                            has_quadratic,
                            has_linear,
                        )

        constraints = []
        for k in range(0, K):
            constraints.append(PSD_constraint_sums[k] >> 0)
            if m_func > 0:
                constraints.append(eq_constraint_sums[k] == 0)

        problem = cp.Problem(cp.Minimize(c_K), constraints)
        solution_handles: _IterationDependentCvxpySolutionHandles = {
            "K": K,
            "dim_Q": dim_Q,
            "dim_q": dim_q,
            "m_func": m_func,
            "Q_0": Q_0,
            "Q_K": Q_K,
            "q_0": q_0,
            "q_K": q_K,
            "Q_vars": Q_vars,
            "q_vars": q_vars,
            "lambdas_op": lambdas_op,
            "lambdas_func": lambdas_func,
            "nus_func": nus_func,
            "c_K_var": c_K,
        }
        return problem, solution_handles

    @staticmethod
    def _pairs_to_readable(pairs: PairTuple) -> List[_ReadablePair]:
        r"""
        Convert internal interpolation pairs to readable dictionary records.

        **Parameters**

        - `pairs` (:class:`~typing.Tuple`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\], ...\]):
          Internal pair tuple.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Union`\[:class:`int`, :class:`str`\]\]\]):
          Readable pair list using keys `"j"` and `"k"`.
        """
        return [{"j": pair[0], "k": pair[1]} for pair in pairs]

    @staticmethod
    def _serialize_iteration_dependent_multipliers(
            multiplier_vars: IterationDependentMultiplierMap,
    ) -> List[_IterationDependentMultiplierRecord]:
        r"""
        Convert scalar multiplier variables into sorted, readable records.

        Sorting is deterministic by `(iteration, component, interpolation_index, pairs)`
        so downstream diagnostics stay stable across runs.

        **Parameters**

        - `multiplier_vars` (:class:`~typing.Dict`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`, :class:`~typing.Tuple`, :class:`int`\], :class:`~typing.Any`\]):
          Internal multiplier-variable mapping.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]\]):
          Sorted readable multiplier records.
        """
        records: List[_IterationDependentMultiplierRecord] = []
        for key, var in sorted(
                multiplier_vars.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][3], str(item[0][2]))):
            iteration, component, pairs, interpolation_index = key
            value = IterationDependent._extract_scalar_variable_value(var)
            records.append({
                "iteration": iteration,
                "component": component,
                "interpolation_index": interpolation_index,
                "pairs": IterationDependent._pairs_to_readable(pairs),
                "value": value,
            })
        return records

    @staticmethod
    def _extract_iteration_dependent_certificate(
            solution_handles: _IterationDependentMosekSolutionHandles,
    ) -> _IterationDependentCertificate:
        r"""
        Extract a solved iteration-dependent certificate into NumPy/Python values.

        **Parameters**

        - `solution_handles` (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]):
          Handle dictionary returned by the MOSEK builder.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]): Certificate with keys
          `Q_sequence`, `q_sequence`, and `multipliers`.
        """
        K = int(solution_handles["K"])
        dim_Q = int(solution_handles["dim_Q"])
        m_func = int(solution_handles["m_func"])
        Q_0 = solution_handles["Q_0"]
        Q_K = solution_handles["Q_K"]
        q_0 = solution_handles["q_0"]
        q_K = solution_handles["q_K"]
        Qij_vars = solution_handles["Qij_vars"]
        q_vars = solution_handles["q_vars"]

        Q_sequence: List[np.ndarray] = []
        for k in range(0, K + 1):
            if k == 0:
                Q_k = np.array(Q_0, dtype=float, copy=True)
            elif k == K:
                Q_k = np.array(Q_K, dtype=float, copy=True)
            else:
                Qij_levels = np.asarray(Qij_vars[k].level(), dtype=float).reshape(-1)
                Q_k = create_symmetric_matrix(Qij_levels, dim_Q)
            Q_sequence.append(Q_k)

        q_sequence: Optional[List[np.ndarray]] = None
        if m_func > 0:
            q_sequence = []
            for k in range(0, K + 1):
                if k == 0:
                    q_k = np.array(q_0, dtype=float, copy=True).reshape(-1)
                elif k == K:
                    q_k = np.array(q_K, dtype=float, copy=True).reshape(-1)
                else:
                    q_k = np.asarray(q_vars[k].level(), dtype=float).reshape(-1)
                q_sequence.append(q_k)

        lambdas_op = solution_handles["lambdas_op"]
        lambdas_func = solution_handles["lambdas_func"]
        nus_func = solution_handles["nus_func"]
        return {
            "Q_sequence": Q_sequence,
            "q_sequence": q_sequence,
            "multipliers": {
                "operator_lambda": IterationDependent._serialize_iteration_dependent_multipliers(lambdas_op),
                "function_lambda": IterationDependent._serialize_iteration_dependent_multipliers(lambdas_func),
                "function_nu": IterationDependent._serialize_iteration_dependent_multipliers(nus_func),
            },
        }

    @staticmethod
    def _extract_iteration_dependent_certificate_cvxpy(
            solution_handles: _IterationDependentCvxpySolutionHandles,
    ) -> _IterationDependentCertificate:
        r"""
        Extract a solved CVXPY-backed iteration-dependent certificate.

        Converts CVXPY variable values to NumPy arrays and preserves the same
        output schema as the MOSEK-based extractor.

        **Parameters**

        - `solution_handles` (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]):
          Handle dictionary returned by the CVXPY builder.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`str`, :class:`~typing.Any`\]): Certificate with keys
          `Q_sequence`, `q_sequence`, and `multipliers`.
        """
        K = int(solution_handles["K"])
        m_func = int(solution_handles["m_func"])
        Q_0 = solution_handles["Q_0"]
        Q_K = solution_handles["Q_K"]
        q_0 = solution_handles["q_0"]
        q_K = solution_handles["q_K"]
        Q_vars = solution_handles["Q_vars"]
        q_vars = solution_handles["q_vars"]

        Q_sequence: List[np.ndarray] = []
        for k in range(0, K + 1):
            if k == 0:
                Q_k = np.array(Q_0, dtype=float, copy=True)
            elif k == K:
                Q_k = np.array(Q_K, dtype=float, copy=True)
            else:
                Q_k = np.asarray(Q_vars[k].value, dtype=float)
            Q_sequence.append(Q_k)

        q_sequence: Optional[List[np.ndarray]] = None
        if m_func > 0:
            q_sequence = []
            for k in range(0, K + 1):
                if k == 0:
                    q_k = np.array(q_0, dtype=float, copy=True).reshape(-1)
                elif k == K:
                    q_k = np.array(q_K, dtype=float, copy=True).reshape(-1)
                else:
                    q_k = np.asarray(q_vars[k].value, dtype=float).reshape(-1)
                q_sequence.append(q_k)

        lambdas_op = solution_handles["lambdas_op"]
        lambdas_func = solution_handles["lambdas_func"]
        nus_func = solution_handles["nus_func"]
        return {
            "Q_sequence": Q_sequence,
            "q_sequence": q_sequence,
            "multipliers": {
                "operator_lambda": IterationDependent._serialize_iteration_dependent_multipliers(lambdas_op),
                "function_lambda": IterationDependent._serialize_iteration_dependent_multipliers(lambdas_func),
                "function_nu": IterationDependent._serialize_iteration_dependent_multipliers(nus_func),
            },
        }

    @staticmethod
    def search_lyapunov(
            prob: InclusionProblem,
            algo: Algorithm,
            K: int,
            Q_0: np.ndarray,
            Q_K: np.ndarray,
            q_0: Optional[np.ndarray] = None,
            q_K: Optional[np.ndarray] = None,
            solver_options: Optional[SolverOptions] = None,
            verbosity: int = 1,
    ) -> Mapping[str, Any]:
        r"""
        Search for an iteration-dependent Lyapunov certificate via an SDP.

        Given an inclusion problem, an algorithm, and user-specified targets
        :math:`(Q_0,q_0,Q_K,q_K,K)`, this method formulates and solves a
        semidefinite feasibility problem for certificate variables
        :math:`(\{Q_k,q_k\}_{k=1}^{K-1}, c_K)`.

        **Connection to theory**

        For the formal statement of the chained quadratic Lyapunov inequality
        and the role of :math:`(Q_0,q_0,Q_K,q_K,K)`, see
        :doc:`/theory/iteration_dependent_analyses`.

        **User-specified targets**

        The tuple :math:`(Q_0,q_0,Q_K,q_K)` fixes the endpoint Lyapunov values
        :math:`\mathcal{V}(Q_0,q_0,0)` and :math:`\mathcal{V}(Q_K,q_K,K)`.
        These endpoint parameters are fixed inputs to the SDP
        (not optimization variables).

        Built-in constructors for choosing endpoint parameters are:

        - :meth:`~autolyap.IterationDependent.get_parameters_distance_to_solution`
        - :meth:`~autolyap.IterationDependent.get_parameters_function_value_suboptimality`
        - :meth:`~autolyap.IterationDependent.get_parameters_fixed_point_residual`
        - :meth:`~autolyap.IterationDependent.get_parameters_optimality_measure`

        The SDP then searches for intermediate :math:`(Q_k,q_k)` for
        :math:`k \in \llbracket 1, K-1\rrbracket` and the minimum feasible
        :math:`c_K \ge 0` so that the chained inequalities hold.
        When :math:`\NumFunc = 0`, the vectors :math:`q_k`
        (and inputs `q_0`, `q_K`) are omitted.

        **Parameters**

        - `prob` (:class:`~typing.Type`\[:class:`~autolyap.problemclass.InclusionProblem`\]): An
          :class:`~autolyap.problemclass.InclusionProblem`
          instance containing interpolation conditions.
        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An
          :class:`~autolyap.algorithms.Algorithm` instance providing
          dimensions and matrix helpers.
        - `K` (:class:`int`): A positive integer corresponding to :math:`K` defining the iteration budget.
        - `Q_0` (:class:`numpy.ndarray`): A symmetric matrix corresponding to
          :math:`Q_0 \in \sym^{n + \NumEval + m}`.
        - `Q_K` (:class:`numpy.ndarray`): A symmetric matrix corresponding to
          :math:`Q_K \in \sym^{n + \NumEval + m}`.
        - `q_0` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): A vector corresponding to
          :math:`q_0 \in \mathbb{R}^{\NumEvalFunc + \NumFunc}` for functional components
          (required if :math:`\NumFunc > 0`; otherwise `None`).
        - `q_K` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): A vector corresponding to
          :math:`q_K \in \mathbb{R}^{\NumEvalFunc + \NumFunc}` for functional components
          (required if :math:`\NumFunc > 0`; otherwise `None`).
        - `solver_options` (:class:`~typing.Optional`\[:class:`~autolyap.solver_options.SolverOptions`\]):
          Optional backend and parameter settings. Defaults to
          `SolverOptions(backend="mosek_fusion")`.
        - `verbosity` (:class:`int`): Nonnegative output level. Defaults to `1`.
          Set `0` to disable user-facing diagnostics, `1` for concise summaries,
          and `2` for per-iteration detail.

        **Returns**

        - (:class:`~typing.Mapping`\[:class:`str`, :class:`~typing.Any`\]): Result mapping
          with keys `status`, `solve_status`, `c_K`, and `certificate`.

          - `status` (:class:`str`): One of `"feasible"`, `"infeasible"`, or
            `"not_solved"`.
          - `solve_status` (:class:`~typing.Optional`\[:class:`str`\]): Raw backend
            solve status (`None` when unavailable).
          - `c_K` (:class:`~typing.Optional`\[:class:`float`\]): Optimal objective value when
            `status == "feasible"`; otherwise `None`. This is the horizon-dependent scalar.
          - `certificate` (:class:`~typing.Optional`\[:class:`~typing.Mapping`\[:class:`str`, :class:`~typing.Any`\]\]):
            `None` unless `status == "feasible"`.

          When `status == "feasible"`, `certificate` has:

          .. code-block:: text

              {
                "Q_sequence": List[np.ndarray],   # length K+1, Q_sequence[k] = Q_k
                "q_sequence": List[np.ndarray] | None,  # None when m_func == 0
                "multipliers": {
                  "operator_lambda": List[record],
                  "function_lambda": List[record],
                  "function_nu": List[record]
                }
              }

          Each multiplier list entry is a `record` with:

          .. code-block:: text

              record = {
                "iteration": int,
                "component": int,
                "interpolation_index": int,
                "pairs": List[{"j": int | "star", "k": int | "star"}],
                "value": float
              }

          Field meanings and ranges:

          - `iteration` corresponds to :math:`k` and satisfies
            :math:`k \in \llbracket 0, K-1 \rrbracket`.
          - `component` corresponds to :math:`i` and satisfies
            :math:`i \in \llbracket 1, m \rrbracket`.
          - `interpolation_index` corresponds to :math:`o`, where
            :math:`o` is the zero-based index from
            :math:`\text{enumerate}(\text{prob.get_component_data}(i))`, so
            :math:`o \in \llbracket 0, \text{len}(\text{prob.get_component_data}(i)) - 1 \rrbracket`.
          - `pairs` is the concrete interpolation-pair list used by that multiplier.
            Its length is determined by
            :class:`~autolyap.problemclass.indices._InterpolationIndices`:
            1 for `"r1"` and 2 for `"r1<r2"`, `"r1!=r2"`, `"r1!=star"`.
            Typical examples are
            `[{"j": 2, "k": 5}]` and
            `[{"j": 1, "k": 5}, {"j": "star", "k": "star"}]`.

            Pair entries satisfy

            .. math::
                \begin{aligned}
                j &\in \llbracket 1, \bar m_i \rrbracket \cup \{\star\}, \\
                k_{\text{pair}} &\in \{k, k+1\} \cup \{\star\},
                \end{aligned}

            with :math:`j=\star \Leftrightarrow k_{\text{pair}}=\star`.
          - `value` is the scalar multiplier for that record.

        **Raises**

        - `ValueError`: If input dimensions or other conditions are violated.
        """
        K, _, _, _, _, m_func, m_op, _, dim_Q, dim_q = IterationDependent._validate_iteration_dependent_inputs(
            prob, algo, K, Q_0, Q_K, q_0, q_K
        )
        verbosity = ensure_integral(verbosity, "verbosity", minimum=0)
        solver_options = _normalize_solver_options(solver_options)

        if solver_options.backend == "mosek_fusion":
            mf = IterationDependent._import_mosek_fusion()
            OptimizeError = mf.OptimizeError
            Mod = mf.Model()
            Mod, solution_handles = IterationDependent._build_iteration_dependent_model(
                prob,
                algo,
                K,
                Q_0,
                Q_K,
                q_0,
                q_K,
                dim_Q,
                dim_q,
                m_func,
                m_op,
                model=Mod,
            )
            IterationDependent._apply_mosek_solver_params(Mod, solver_options)
            c_K_var = solution_handles["c_K_var"]
            if verbosity > 0:
                print(
                    f"[AutoLyap][INFO] Solving iteration-dependent SDP "
                    f"(backend={solver_options.backend}, K={K})."
                )

            try:
                Mod.solve()
                status = Mod.getProblemStatus()
                solve_status = str(status)
                classified_status = _classify_mosek_problem_status(status)
                if classified_status == _RESULT_STATUS_INFEASIBLE:
                    if verbosity > 0:
                        print(
                            f"[AutoLyap][INFO] Iteration-dependent SDP status={status}; "
                            f"no feasible certificate for K={K}."
                        )
                    return _make_iteration_dependent_result(
                        status=_RESULT_STATUS_INFEASIBLE,
                        solve_status=solve_status,
                        c_K=None,
                        certificate=None,
                    )
                if classified_status == _RESULT_STATUS_NOT_SOLVED:
                    if verbosity > 0:
                        print(
                            f"[AutoLyap][INFO] Iteration-dependent SDP status={status}; "
                            f"solver did not return a feasibility certificate for K={K}."
                        )
                    return _make_iteration_dependent_result(
                        status=_RESULT_STATUS_NOT_SOLVED,
                        solve_status=solve_status,
                        c_K=None,
                        certificate=None,
                    )
                c_K_val = IterationDependent._extract_scalar_variable_value(c_K_var)
                certificate = IterationDependent._extract_iteration_dependent_certificate(solution_handles)
                if verbosity > 0:
                    try:
                        diagnostics = IterationDependent._compute_iteration_dependent_diagnostics(
                            prob,
                            algo,
                            K,
                            c_K_val,
                            certificate,
                        )
                        IterationDependent._print_iteration_dependent_diagnostics(
                            diagnostics,
                            K,
                            c_K_val,
                            solver_options.backend,
                            verbosity,
                        )
                    except Exception as exc:
                        print(
                            f"[AutoLyap][WARN] Unable to compute diagnostic summary: {exc}."
                        )
                return _make_iteration_dependent_result(
                    status=_RESULT_STATUS_FEASIBLE,
                    solve_status=solve_status,
                    c_K=c_K_val,
                    certificate=certificate,
                )
            except OptimizeError as e:
                if _is_mosek_license_error(e):
                    raise
                if verbosity > 0:
                    print(
                        f"[AutoLyap][INFO] Iteration-dependent SDP solve failed for K={K}: {e}."
                    )
                return _make_iteration_dependent_result(
                    status=_RESULT_STATUS_NOT_SOLVED,
                    solve_status="optimize_error",
                    c_K=None,
                    certificate=None,
                )
            finally:
                Mod.dispose()

        cp = IterationDependent._import_cvxpy()
        problem, cvxpy_solution_handles = IterationDependent._build_iteration_dependent_problem_cvxpy(
            prob,
            algo,
            K,
            Q_0,
            Q_K,
            q_0,
            q_K,
            dim_Q,
            dim_q,
            m_func,
            m_op,
            cp=cp,
        )
        solve_kwargs = _get_cvxpy_solve_kwargs(solver_options)
        accepted_statuses = _get_cvxpy_accepted_statuses(cp, solver_options)
        cvxpy_solver_error = getattr(getattr(cp, "error", None), "SolverError", None)
        if verbosity > 0:
            print(
                f"[AutoLyap][INFO] Solving iteration-dependent SDP "
                f"(backend={solver_options.backend}, K={K})."
            )
        try:
            problem.solve(**solve_kwargs)
        except Exception as exc:
            if cvxpy_solver_error is not None and isinstance(exc, cvxpy_solver_error):
                if verbosity > 0:
                    print(
                        f"[AutoLyap][INFO] Iteration-dependent SDP solver error for K={K}: {exc}."
                    )
                return _make_iteration_dependent_result(
                    status=_RESULT_STATUS_NOT_SOLVED,
                    solve_status="solver_error",
                    c_K=None,
                    certificate=None,
                )
            raise

        classified_status = _classify_cvxpy_problem_status(problem.status, cp, accepted_statuses)
        solve_status = str(problem.status)
        if classified_status == _RESULT_STATUS_INFEASIBLE:
            if verbosity > 0:
                print(
                    f"[AutoLyap][INFO] Iteration-dependent SDP status={problem.status}; no feasible certificate for K={K}."
                )
            return _make_iteration_dependent_result(
                status=_RESULT_STATUS_INFEASIBLE,
                solve_status=solve_status,
                c_K=None,
                certificate=None,
            )
        if classified_status == _RESULT_STATUS_NOT_SOLVED:
            if verbosity > 0:
                print(
                    f"[AutoLyap][INFO] Iteration-dependent SDP status={problem.status}; "
                    f"solver did not return a feasibility certificate for K={K}."
                )
            return _make_iteration_dependent_result(
                status=_RESULT_STATUS_NOT_SOLVED,
                solve_status=solve_status,
                c_K=None,
                certificate=None,
            )

        c_K_val = IterationDependent._extract_scalar_variable_value(cvxpy_solution_handles["c_K_var"])
        certificate = IterationDependent._extract_iteration_dependent_certificate_cvxpy(cvxpy_solution_handles)
        if verbosity > 0:
            try:
                diagnostics = IterationDependent._compute_iteration_dependent_diagnostics(
                    prob,
                    algo,
                    K,
                    c_K_val,
                    certificate,
                )
                IterationDependent._print_iteration_dependent_diagnostics(
                    diagnostics,
                    K,
                    c_K_val,
                    solver_options.backend,
                    verbosity,
                )
            except Exception as exc:
                print(
                    f"[AutoLyap][WARN] Unable to compute diagnostic summary: {exc}."
                )
        return _make_iteration_dependent_result(
            status=_RESULT_STATUS_FEASIBLE,
            solve_status=solve_status,
            c_K=c_K_val,
            certificate=certificate,
        )

    @staticmethod
    def _compute_Thetas(algo: Algorithm, k: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the capital :math:`\Theta` matrices for the iteration-dependent Lyapunov context.

        The matrices are defined as follows:

        .. math::

            \Theta_{0} =
            \begin{bmatrix}
            I_{n+\NumEval} & 0_{(n+\NumEval)\times\NumEval} & 0_{(n+\NumEval)\times m} \\
            0_{m\times(n+\NumEval)} & 0_{m\times\NumEval} & I_{m}
            \end{bmatrix}

        and

        .. math::

            \Theta_{1}^{(k)} =
            \begin{bmatrix}
            X_{k+1}^{k,k+1} \\
            0_{(\NumEval+m)\times(n+\NumEval)} \quad I_{(\NumEval+m)}
            \end{bmatrix}

        **Definitions**

        - :math:`n` is given by `algo.n`.
        - :math:`\NumEval` is given by `algo.m_bar`.
        - :math:`m` is given by `algo.m`.
        - :math:`X_{k+1}^{k,k+1}` is retrieved via :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Xs` with
          `k_min = k` and `k_max = k+1`, using key :math:`k+1`.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`.
        - `k` (:class:`int`): A non-negative integer iteration index corresponding to :math:`k` used to select
          the appropriate :math:`X` matrix.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(\Theta_0, \Theta_1)` of numpy arrays.

        **Raises**

        - `ValueError`: If :math:`k` is negative or the required :math:`X` matrix is missing.
        """

        k = ensure_integral(k, "k", minimum=0)

        n = algo.n
        m_bar = algo.m_bar
        m = algo.m

        # Construct Theta_{0}
        Theta0 = np.block([
            [np.eye(n + m_bar), np.zeros((n + m_bar, m_bar)), np.zeros((n + m_bar, m))],
            [np.zeros((m, n + m_bar)), np.zeros((m, m_bar)), np.eye(m)]
        ])

        # Retrieve X_{k+1}^{k,k+1} using algo._get_Xs(k, k+1)
        Xs = algo._get_Xs(k, k+1)
        if (k + 1) not in Xs:
            raise ValueError(f"Expected key {k+1} in X matrices, but it was not found.")
        X_block = Xs[k + 1]

        # Construct the lower block for Theta_{1}^{(k)}
        lower_block = np.hstack([
            np.zeros((m_bar + m, n + m_bar)),
            np.eye(m_bar + m)
        ])

        # Form Theta_{1}^{(k)} by vertically stacking X_block and the lower block
        Theta1 = np.vstack([X_block, lower_block])
        return Theta0, Theta1

    @staticmethod
    def _compute_thetas(algo: Algorithm) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the lowercase :math:`\theta` matrices for the iteration-dependent Lyapunov context.

        The matrices are defined as follows:

        .. math::

            \theta_{0} =
            \begin{bmatrix}
            I_{\NumEvalFunc} & 0_{\NumEvalFunc \times \NumEvalFunc} & 0_{\NumEvalFunc \times \NumFunc} \\
            0_{\NumFunc \times \NumEvalFunc} & 0_{\NumFunc \times \NumEvalFunc} & I_{\NumFunc}
            \end{bmatrix}

        and

        .. math::

            \theta_{1} =
            \begin{bmatrix}
            0_{(\NumEvalFunc+\NumFunc) \times \NumEvalFunc} & I_{(\NumEvalFunc+\NumFunc)}
            \end{bmatrix}

        **Definitions**

        - :math:`\NumEvalFunc` is given by `algo.m_bar_func`.
        - :math:`\NumFunc` is given by `algo.m_func`.

        **Notes**

        - The :math:`\theta` matrices are only defined when there is at least one functional component.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(\theta_{0}, \theta_{1})` of numpy arrays.

        **Raises**

        - `ValueError`: If there are no functional components.
        """
        m_bar_func = algo.m_bar_func
        m_func = algo.m_func
        if m_func <= 0:
            raise ValueError("Theta matrices require at least one functional component (m_func > 0).")

        theta0 = np.block([
            [np.eye(m_bar_func), np.zeros((m_bar_func, m_bar_func)), np.zeros((m_bar_func, m_func))],
            [np.zeros((m_func, m_bar_func)), np.zeros((m_func, m_bar_func)), np.eye(m_func)]
        ])

        theta1 = np.hstack([
            np.zeros((m_bar_func + m_func, m_bar_func)),
            np.eye(m_bar_func + m_func)
        ])

        return theta0, theta1

    @staticmethod
    def get_parameters_distance_to_solution(
            algo: Algorithm,
            k: int,
            i: int = 1,
            j: int = 1
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute Lyapunov parameters for the distance-to-solution metric at iteration :math:`k`.

        For the matrix constructions used in this method, see
        :doc:`/theory/performance_estimation_via_sdps`.
        For the role of :math:`(Q_k,q_k)`, see
        :doc:`/theory/iteration_dependent_analyses`.

        **Resulting lower bounds**

        With this choice of :math:`(Q_k,q_k)`,

        .. math::
            \mathcal{V}(Q_k,q_k,k) = \|y_{i,j}^{k} - y^{\star}\|^{2}.

        **Matrix construction**

        The matrix :math:`Q_k` is constructed as

        .. math::
            Q_k
            =
            \left(P_{(i,j)}Y_k^{k,k} - P_{(i,\star)}Y_\star^{k,k}\right)^\top
            \left(P_{(i,j)}Y_k^{k,k} - P_{(i,\star)}Y_\star^{k,k}\right),

        where:

        - :math:`Y_k^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ys`, using `k_min = k` and `k_max = k`.
        - :math:`Y_\star^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ys`, using `k_min = k` and `k_max = k`.
        - :math:`P_{(i,j)}` and :math:`P_{(i,\star)}` are projection matrices retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ps`.

        The remaining vector is set to zero:

        - If :math:`\NumFunc > 0`, then :math:`q_k = 0`.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): Algorithm instance providing `m`, `m_bar_is`,
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ys`, and
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ps`.
        - `k` (:class:`int`): Nonnegative iteration index :math:`k`.
        - `i` (:class:`int`): Component index, with
          :math:`i \in \llbracket 1, m\rrbracket`.
        - `j` (:class:`int`): Evaluation index for component `i`, with
          :math:`j \in \llbracket 1, \NumEval_i\rrbracket`.

        **Returns**

        - (:class:`~typing.Union`\[:class:`numpy.ndarray`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns :math:`Q_k` with

          .. math::
              Q_k \in \sym^{n + \NumEval + m}.

          Otherwise, returns :math:`(Q_k, q_k)` with

          .. math::
              \begin{aligned}
              Q_k &\in \sym^{n + \NumEval + m},\\
              q_k &\in \mathbb{R}^{\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If an input is invalid or a required matrix is missing.
        """
        # ----- Input Checking -----
        k = ensure_integral(k, "k", minimum=0)
        i = ensure_integral(i, "i", minimum=1)
        if i > algo.m:
            raise ValueError(f"Component index i must be in [1, {algo.m}]. Got {i}.")
        j = ensure_integral(j, "j", minimum=1)
        if j > algo.m_bar_is[i - 1]:
            raise ValueError(f"For component {i}, evaluation index j must be in [1, {algo.m_bar_is[i - 1]}]. Got {j}.")

        # ----- Compute Q_k -----
        Ys = algo._get_Ys(k, k)
        if k not in Ys:
            raise ValueError(f"Y matrix for iteration k = {k} not found.")
        if 'star' not in Ys:
            raise ValueError("Y star matrix ('star') not found.")

        Ps = algo._get_Ps()
        if (i, j) not in Ps:
            raise ValueError(f"Projection matrix for component {i}, evaluation {j} not found.")
        if (i, 'star') not in Ps:
            raise ValueError(f"Projection matrix for component {i} star not found.")

        diff = Ps[(i, j)] @ Ys[k] - Ps[(i, 'star')] @ Ys['star']
        Q_k = diff.T @ diff

        # ----- Construct T, p, and t as zeros with appropriate dimensions -----
        if algo.m_func > 0:
            m_bar_func = algo.m_bar_func    # Total evaluations for functional components.
            m_func = algo.m_func            # Number of functional components.
            dim_q_k = m_bar_func + m_func
            q_k_vec = np.zeros(dim_q_k)
            return Q_k, q_k_vec
        else:
            return Q_k

    @staticmethod
    def get_parameters_function_value_suboptimality(
            algo: Algorithm,
            k: int,
            j: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute Lyapunov parameters for function-value suboptimality at iteration :math:`k`.

        For the matrix constructions used in this method, see
        :doc:`/theory/performance_estimation_via_sdps`.
        For the role of :math:`(Q_k,q_k)`, see
        :doc:`/theory/iteration_dependent_analyses`.

        **Resulting lower bounds**

        With this choice of :math:`(Q_k,q_k)`,

        .. math::
            \mathcal{V}(Q_k,q_k,k) = f_{1}(y_{1,j}^{k}) - f_{1}(y^{\star}).

        **Matrix construction**

        This method applies only when :math:`m = \NumFunc = 1`.

        The vector :math:`q_k` is constructed as

        .. math::
            q_k = \left(F_{(1,j,k)}^{k,k} - F_{(1,\star,\star)}^{k,k}\right)^\top,

        where:

        - :math:`F_{(1,j,k)}^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Fs`, using `k_min = k` and `k_max = k`.
        - :math:`F_{(1,\star,\star)}^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Fs`, using `k_min = k` and `k_max = k`.

        The remaining matrix is set to zero:

        - :math:`Q_k = 0`.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): Algorithm instance with `m = m_func = 1`,
          `m_bar_is`, and :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Fs`.
        - `k` (:class:`int`): Nonnegative iteration index :math:`k`.
        - `j` (:class:`int`): Evaluation index for component 1, with
          :math:`j \in \llbracket 1, \NumEval_1\rrbracket`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]):
          A tuple :math:`(Q_k, q_k)` with

          .. math::
              \begin{aligned}
              Q_k &\in \sym^{n + \NumEval + m},\\
              q_k &\in \mathbb{R}^{\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If an input is invalid, required :math:`F` matrices are missing,
          or :math:`m \ne 1` / :math:`\NumFunc \ne 1`.
        """
        # ----- Input Checking -----
        k = ensure_integral(k, "k", minimum=0)
        j = ensure_integral(j, "j", minimum=1)
        if j > algo.m_bar_is[0]:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {algo.m_bar_is[0]}]. Got {j}.")
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("Function value suboptimality is defined only for problems with a single functional component (m = m_func = 1).")

        # ----- Dimensions for Q_k -----
        dim_Q_k = algo.n + algo.m_bar + algo.m
        Q_k = np.zeros((dim_Q_k, dim_Q_k))

        # ----- Compute q_k -----
        Fs = algo._get_Fs(k, k)
        key_nonstar = (1, j, k)
        key_star = (1, 'star', 'star')
        if key_nonstar not in Fs:
            raise ValueError(f"F matrix for key {key_nonstar} not found.")
        if key_star not in Fs:
            raise ValueError("F star matrix (1, 'star', 'star') not found.")
        F_nonstar = Fs[key_nonstar]
        F_star = Fs[key_star]
        q_k = (F_nonstar - F_star).T
        q_k = np.ravel(q_k)

        return Q_k, q_k

    @staticmethod
    def get_parameters_fixed_point_residual(
            algo: Algorithm,
            k: int
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute Lyapunov parameters for the fixed-point residual at iteration :math:`k`.

        For the matrix constructions used in this method, see
        :doc:`/theory/performance_estimation_via_sdps`.
        For the role of :math:`(Q_k,q_k)`, see
        :doc:`/theory/iteration_dependent_analyses`.

        **Resulting lower bounds**

        With this choice of :math:`(Q_k,q_k)`,

        .. math::
            \mathcal{V}(Q_k,q_k,k) = \|\bx^{k+1} - \bx^{k}\|^{2}.

        **Matrix construction**

        The matrix :math:`Q_k` is constructed as

        .. math::
            Q_k
            =
            \left(X_{k+1}^{k,k} - X_k^{k,k}\right)^\top
            \left(X_{k+1}^{k,k} - X_k^{k,k}\right),

        where:

        - :math:`X_k^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Xs`, using `k_min = k` and `k_max = k`.
        - :math:`X_{k+1}^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Xs`, using `k_min = k` and `k_max = k`.

        The remaining vector is set to zero:

        - If :math:`\NumFunc > 0`, then :math:`q_k = 0`.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): Algorithm instance providing dimensions and
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Xs`.
        - `k` (:class:`int`): Nonnegative iteration index :math:`k`.

        **Returns**

        - (:class:`~typing.Union`\[:class:`numpy.ndarray`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns :math:`Q_k` with

          .. math::
              Q_k \in \sym^{n + \NumEval + m}.

          Otherwise, returns :math:`(Q_k, q_k)` with

          .. math::
              \begin{aligned}
              Q_k &\in \sym^{n + \NumEval + m},\\
              q_k &\in \mathbb{R}^{\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If an input is invalid or required :math:`X` matrices are missing.
        """
        # ----- Input Checking -----
        k = ensure_integral(k, "k", minimum=0)

        # ----- Retrieve X matrices -----
        Xs = algo._get_Xs(k, k)
        if k not in Xs or (k + 1) not in Xs:
            raise ValueError(f"X matrices for iterations {k} and {k+1} not found.")

        # ----- Compute Q_k -----
        diff = Xs[k + 1] - Xs[k]
        Q_k = diff.T @ diff

        # ----- Set q_k to zero -----
        if algo.m_func > 0:
            q_dim = algo.m_bar_func + algo.m_func
            q_k = np.zeros(q_dim)
            return Q_k, q_k
        else:
            return Q_k

    @staticmethod
    def get_parameters_optimality_measure(
            algo: Algorithm,
            k: int
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute Lyapunov parameters for the optimality measure at iteration :math:`k`.

        For the matrix constructions used in this method, see
        :doc:`/theory/performance_estimation_via_sdps`.
        For the role of :math:`(Q_k,q_k)`, see
        :doc:`/theory/iteration_dependent_analyses`.

        **Resulting lower bounds**

        With this choice of :math:`(Q_k,q_k)`,

        .. math::
            \mathcal{V}(Q_k,q_k,k)
            =
            \begin{cases}
                \|u_{1,1}^{k}\|^{2}, & \text{if } m = 1, \\[0.5em]
                \left\|\sum_{i=1}^{m} u_{i,1}^{k}\right\|^{2}
                + \sum_{i=2}^{m} \|y_{1,1}^{k} - y_{i,1}^{k}\|^{2},
                & \text{if } m > 1.
            \end{cases}

        **Matrix construction**

        The matrix :math:`Q_k` is constructed as

        .. math::
            Q_k
            =
            \begin{cases}
                \left(P_{(1,1)}U_k^{k,k}\right)^\top \left(P_{(1,1)}U_k^{k,k}\right), & \text{if } m = 1, \\[1em]
                \left(\sum_{i=1}^{m} P_{(i,1)}U_k^{k,k}\right)^\top
                \left(\sum_{i=1}^{m} P_{(i,1)}U_k^{k,k}\right)
                + \sum_{i=2}^{m}
                \left(\left(P_{(1,1)} - P_{(i,1)}\right)Y_k^{k,k}\right)^\top
                \left(\left(P_{(1,1)} - P_{(i,1)}\right)Y_k^{k,k}\right),
                & \text{if } m > 1,
            \end{cases}

        where:

        - :math:`U_k^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Us`, using `k_min = k` and `k_max = k`.
        - :math:`Y_k^{k,k}` is retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ys`, using `k_min = k` and `k_max = k`.
        - :math:`P_{(i,1)}` are projection matrices retrieved via
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ps`.

        The remaining vector is set to zero:

        - If :math:`\NumFunc > 0`, then :math:`q_k = 0`.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): Algorithm instance providing `m`, `m_bar`, `m_bar_is`,
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Us`,
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ys`, and
          :meth:`~autolyap.algorithms.algorithm.Algorithm._get_Ps`.
        - `k` (:class:`int`): Nonnegative iteration index :math:`k`.

        **Returns**

        - (:class:`~typing.Union`\[:class:`numpy.ndarray`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns :math:`Q_k` with

          .. math::
              Q_k \in \sym^{n + \NumEval + m}.

          Otherwise, returns :math:`(Q_k, q_k)` with

          .. math::
              \begin{aligned}
              Q_k &\in \sym^{n + \NumEval + m},\\
              q_k &\in \mathbb{R}^{\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If an input is invalid or required matrices are missing.
        """
        # ----- Input Checking -----
        k = ensure_integral(k, "k", minimum=0)

        # ----- Retrieve U and Y matrices -----
        Us = algo._get_Us(k, k)
        Ys = algo._get_Ys(k, k)
        if k not in Us:
            raise ValueError(f"U matrix for iteration {k} not found.")
        if k not in Ys:
            raise ValueError(f"Y matrix for iteration {k} not found.")

        # ----- Retrieve Projection matrices -----
        Ps = algo._get_Ps()

        # ----- Compute Q_k -----
        if algo.m == 1:
            # Q_k = (P_{(1,1)}U_{k}^{k,k})^{\top} P_{(1,1)}U_{k}^{k,k}
            if algo.m_bar_is[0] < 1:
                raise ValueError("Optimality measure requires at least one evaluation for component 1.")
            P = Ps[(1, 1)]
            U = Us[k]
            term = P @ U
            Q_k = term.T @ term
        else:
            if any(m_bar_i < 1 for m_bar_i in algo.m_bar_is):
                raise ValueError("Optimality measure requires each component to have at least one evaluation.")
            # m > 1:
            # term1 = (sum_{i=1}^{m} P_{(i,1)}U_{k}^{k,k})^{\top} (sum_{i=1}^{m} P_{(i,1)}U_{k}^{k,k})
            S = np.zeros((1, Us[k].shape[1]))
            for i in range(1, algo.m + 1):
                P_i = Ps[(i, 1)]
                S = S + P_i @ Us[k]
            term1 = S.T @ S

            # term2 = sum_{i=2}^{m} ((P_{(1,1)} - P_{(i,1)})Y_{k}^{k,k})^{\top} ((P_{(1,1)} - P_{(i,1)})Y_{k}^{k,k})
            Y = Ys[k]
            term2 = 0
            for i in range(2, algo.m + 1):
                diff_P = Ps[(1, 1)] - Ps[(i, 1)]
                temp = diff_P @ Y
                term2 = term2 + (temp.T @ temp)
            Q_k = term1 + term2

        # ----- Set q_k to zero -----
        if algo.m_func > 0:
            q_dim = algo.m_bar_func + algo.m_func
            q_k = np.zeros(q_dim)
            return Q_k, q_k
        else:
            return Q_k
