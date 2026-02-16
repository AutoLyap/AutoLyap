#!/usr/bin/env python3
"""Run and sanity-check verbosity output for AutoLyap Lyapunov solvers."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, Tuple

from autolyap import IterationDependent, IterationIndependent
from autolyap.algorithms import GradientMethod, OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex, SmoothStronglyConvex
from autolyap.solver_options import SolverOptions


def _build_solver_options(args: argparse.Namespace) -> SolverOptions:
    if args.backend == "mosek_fusion":
        return SolverOptions(backend="mosek_fusion")

    cvxpy_solver = args.cvxpy_solver if args.cvxpy_solver else None
    cvxpy_params: Dict[str, object] = {"verbose": False}
    if args.max_iter is not None and cvxpy_solver is not None:
        if cvxpy_solver.upper() == "SCS":
            cvxpy_params["max_iters"] = args.max_iter
        else:
            cvxpy_params["max_iter"] = args.max_iter
    elif args.max_iter is not None and cvxpy_solver is None:
        print(
            "[SCRIPT][WARN] --max-iter was ignored because --cvxpy-solver was not set.",
            file=sys.stderr,
        )
    return SolverOptions(
        backend="cvxpy",
        cvxpy_solver=cvxpy_solver,
        cvxpy_solver_params=cvxpy_params,
    )


def _run_verify_iteration_independent(
    solver_options: SolverOptions, verbosity: int
) -> Dict[str, object]:
    problem = InclusionProblem([SmoothStronglyConvex(mu=1.0, L=4.0)])
    algorithm = GradientMethod(gamma=0.2)
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
    )
    return IterationIndependent.search_lyapunov(
        problem,
        algorithm,
        P,
        T,
        p=p,
        t=t,
        rho=1.0,
        solver_options=solver_options,
        verbosity=verbosity,
    )


def _run_bisection_iteration_independent(
    solver_options: SolverOptions, verbosity: int
) -> Dict[str, object]:
    problem = InclusionProblem([SmoothStronglyConvex(mu=1.0, L=4.0)])
    algorithm = GradientMethod(gamma=0.2)
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
    )
    return IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        p=p,
        t=t,
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-4,
        solver_options=solver_options,
        verbosity=verbosity,
    )


def _run_verify_iteration_dependent(
    solver_options: SolverOptions, verbosity: int
) -> Dict[str, object]:
    K = 3
    problem = InclusionProblem([SmoothConvex(1.0)])
    algorithm = OptimizedGradientMethod(L=1.0, K=K)
    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(algorithm, k=0, i=1, j=1)
    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(algorithm, k=K)
    return IterationDependent.search_lyapunov(
        problem,
        algorithm,
        K,
        Q_0,
        Q_K,
        q_0=q_0,
        q_K=q_K,
        solver_options=solver_options,
        verbosity=verbosity,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all AutoLyap verbosity modes (0/1/2) for iteration-independent "
            "verification, iteration-independent bisection, and iteration-dependent verification."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("cvxpy", "mosek_fusion"),
        default="cvxpy",
        help="Solver backend to use (default: cvxpy).",
    )
    parser.add_argument(
        "--cvxpy-solver",
        default="",
        help='Optional CVXPY solver name (for example: "CLARABEL", "SCS", "MOSEK").',
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional max_iter forwarded to CVXPY solver params.",
    )
    args = parser.parse_args()
    solver_options = _build_solver_options(args)

    cases: Tuple[Tuple[str, Callable[[SolverOptions, int], Dict[str, object]], str], ...] = (
        (
            "iteration_independent.search_lyapunov",
            _run_verify_iteration_independent,
            "rho",
        ),
        (
            "bisection_search_rho",
            _run_bisection_iteration_independent,
            "rho",
        ),
        (
            "iteration_dependent.search_lyapunov",
            _run_verify_iteration_dependent,
            "c",
        ),
    )

    failures = 0
    for case_name, case_runner, scalar_key in cases:
        for verbosity in (0, 1, 2):
            print()
            print("=" * 80)
            print(f"Case: {case_name} | verbosity={verbosity} | backend={solver_options.backend}")
            print("=" * 80)
            try:
                result = case_runner(solver_options, verbosity)
                status = str(result.get("status", ""))
                feasible = status == "feasible"
                scalar_value = result.get(scalar_key, None)
                print(
                    f"[SCRIPT] Result: status={status}, {scalar_key}={scalar_value}, "
                    f"certificate_present={result.get('certificate') is not None}"
                )
                if not feasible:
                    failures += 1
                    print(
                        f"[SCRIPT][WARN] {case_name} returned status={status!r} at verbosity={verbosity}."
                    )
            except Exception as exc:
                failures += 1
                print(f"[SCRIPT][ERROR] {case_name} raised an exception at verbosity={verbosity}: {exc}")

    print()
    print("-" * 80)
    if failures == 0:
        print("[SCRIPT] Completed successfully. All runs returned status='feasible'.")
        return 0

    print(f"[SCRIPT] Completed with {failures} failing run(s).")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
