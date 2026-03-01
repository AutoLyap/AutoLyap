#!/usr/bin/env python3
"""Generate linear Chambolle--Pock paper data by direct solves on the paper grid.

This script computes the smallest certifiable linear contraction factor rho at each
grid point (tau=sigma, theta), using AutoLyap directly on the ver_6 grid:
  - tau = sigma in linspace(1.0, 1.8, 96)
  - theta in linspace(0.0, 1.5, 48)

Output:
  github/paper_data/linear_chambolle_pock/chambolle_pock_smooth_strongly_convex_tau_theta_rho.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Sequence


# Allow running from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import ChambollePock
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex


DEFAULT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        raise ValueError(f"Expected n >= 2, got {n}.")
    step = (b - a) / float(n - 1)
    return [a + i * step for i in range(n)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute linear Chambolle--Pock rho values directly on the ver_6 paper grid."
        )
    )
    parser.add_argument("--tau-min", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=1.8)
    parser.add_argument("--tau-count", type=int, default=96)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=1.5)
    parser.add_argument("--theta-count", type=int, default=48)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--L", type=float, default=50.0)
    parser.add_argument(
        "--backend",
        choices=("mosek_fusion", "cvxpy"),
        default="mosek_fusion",
    )
    parser.add_argument("--lower-bound", type=float, default=0.0)
    parser.add_argument("--upper-bound", type=float, default=1.0)
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Bisection tolerance. If omitted, use AutoLyap's internal default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_data" / "linear_chambolle_pock",
    )
    parser.add_argument(
        "--output-name",
        default="chambolle_pock_smooth_strongly_convex_tau_theta_rho.tex",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.tau_count < 2:
        raise ValueError(f"tau-count must be >= 2, got {args.tau_count}.")
    if args.theta_count < 2:
        raise ValueError(f"theta-count must be >= 2, got {args.theta_count}.")
    if args.tau_max < args.tau_min:
        raise ValueError(f"tau-max must be >= tau-min, got {args.tau_max} < {args.tau_min}.")
    if args.theta_max < args.theta_min:
        raise ValueError(
            f"theta-max must be >= theta-min, got {args.theta_max} < {args.theta_min}."
        )
    if not (0.0 < args.mu < args.L):
        raise ValueError(f"Require 0 < mu < L. Got mu={args.mu}, L={args.L}.")
    if args.lower_bound < 0.0:
        raise ValueError(f"lower-bound must be >= 0, got {args.lower_bound}.")
    if args.upper_bound < args.lower_bound:
        raise ValueError(
            f"upper-bound must be >= lower-bound, got {args.upper_bound} < {args.lower_bound}."
        )
    if args.tol is not None and args.tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {args.tol}.")


def _make_solver_options(backend: str) -> SolverOptions:
    if backend == "mosek_fusion":
        return SolverOptions(backend="mosek_fusion", mosek_params=DEFAULT_MOSEK_PARAMS)
    return SolverOptions(backend="cvxpy")


def _run_scan(
    tau_grid: Sequence[float],
    theta_grid: Sequence[float],
    args: argparse.Namespace,
) -> tuple[list[tuple[float, float, float]], int]:
    problem = InclusionProblem(
        [
            SmoothStronglyConvex(mu=float(args.mu), L=float(args.L)),
            SmoothStronglyConvex(mu=float(args.mu), L=float(args.L)),
        ]
    )
    algorithm = ChambollePock(
        tau=float(tau_grid[0]),
        sigma=float(tau_grid[0]),
        theta=float(theta_grid[0]),
    )
    solver_options = _make_solver_options(args.backend)

    rows: list[tuple[float, float, float]] = []
    errors = 0
    processed = 0
    total = len(theta_grid) * len(tau_grid)

    for row_idx, theta in enumerate(theta_grid, start=1):
        row_feasible = 0
        theta_float = float(theta)

        for tau in tau_grid:
            processed += 1
            tau_float = float(tau)
            algorithm.set_tau(tau_float)
            algorithm.set_sigma(tau_float)
            algorithm.set_theta(theta_float)

            P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                algorithm,
                i=1,
            )

            try:
                bisection_kwargs = {
                    "prob": problem,
                    "algo": algorithm,
                    "P": P,
                    "T": T,
                    "p": p,
                    "t": t,
                    "lower_bound": float(args.lower_bound),
                    "upper_bound": float(args.upper_bound),
                    "solver_options": solver_options,
                    "verbosity": 0,
                }
                if args.tol is not None:
                    bisection_kwargs["tol"] = float(args.tol)
                result = IterationIndependent.LinearConvergence.bisection_search_rho(
                    **bisection_kwargs
                )
            except Exception as exc:
                errors += 1
                print(
                    f"[scan] error at tau=sigma={tau_float:.6f}, theta={theta_float:.6f}: {exc}"
                )
                continue

            if result.get("status") != "feasible":
                continue

            rho_value = result.get("rho")
            if rho_value is None:
                continue
            rho_float = float(rho_value)
            if not math.isfinite(rho_float):
                continue

            rows.append((tau_float, theta_float, rho_float))
            row_feasible += 1

        print(
            f"[scan] row {row_idx:>2}/{len(theta_grid)} theta={theta_float:.6f} "
            f"feasible={row_feasible:>3}/{len(tau_grid)} cumulative={len(rows):>5}/{processed:>5} "
            f"total={total}"
        )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["tau", "theta", "rho"])
        for tau, theta, rho in rows:
            writer.writerow([repr(float(tau)), repr(float(theta)), repr(float(rho))])


def main() -> int:
    args = _build_parser().parse_args()
    _validate(args)

    tau_grid = _linspace(args.tau_min, args.tau_max, args.tau_count)
    theta_grid = _linspace(args.theta_min, args.theta_max, args.theta_count)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}, L={args.L}")
    print(
        f"Grid: tau=sigma in [{tau_grid[0]:.6f}, {tau_grid[-1]:.6f}] ({len(tau_grid)} points), "
        f"theta in [{theta_grid[0]:.6f}, {theta_grid[-1]:.6f}] ({len(theta_grid)} points)"
    )
    tol_msg = args.tol if args.tol is not None else "AutoLyap default"
    print(f"Bisection: lower_bound={args.lower_bound}, upper_bound={args.upper_bound}, tol={tol_msg}")
    print()

    started = time.time()
    rows, errors = _run_scan(tau_grid, theta_grid, args)

    out_path = args.output_dir / args.output_name
    _write_rows(out_path, rows)

    elapsed = time.time() - started
    print()
    print(f"Feasible points: {len(rows)}")
    print(f"Solver errors:   {errors}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print(f"Wrote:           {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
