#!/usr/bin/env python3
"""Generate paper data tables for the Chambolle--Pock plot.

This writes the four CSV-style `.tex` tables consumed by:
  overleaf/ver_6/plots/chambolle_pock.tex

Outputs are written under:
  github/paper_data/sublinear_chambolle_pock/

Sweep:
  - tau = sigma in linspace(1.0, 1.8, 96)
  - theta in linspace(0.0, 1.5, 48)
  - (h, alpha) in {(0,0), (1,0), (1,1), (2,0)}
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import ChambollePock
from autolyap.problemclass import Convex, InclusionProblem


LAYER_ORDER = ((0, 0), (1, 0), (1, 1), (2, 0))
DEFAULT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-6,
    "intpntCoTolDfeas": 1e-6,
    "intpntCoTolRelGap": 1e-6,
    "intpntMaxIterations": 10000,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper_data tables for the Chambolle--Pock layered region plot."
    )
    parser.add_argument("--tau-min", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=1.8)
    parser.add_argument("--tau-count", type=int, default=96)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=1.5)
    parser.add_argument("--theta-count", type=int, default=48)
    parser.add_argument(
        "--backend",
        choices=("mosek_fusion", "cvxpy"),
        default="mosek_fusion",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_data" / "sublinear_chambolle_pock",
        help="Directory where output tables are written.",
    )
    parser.add_argument(
        "--parallel-layers",
        action="store_true",
        help="Sweep each (h, alpha) layer in a separate worker process.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum workers used with --parallel-layers.",
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


def _output_path(out_dir: Path, h: int, alpha: int) -> Path:
    return out_dir / f"chambolle_pock_fixed_point_residual_h{h}_alpha{alpha}.tex"


def _write_xy(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["x", "y"])
        for x, y in rows:
            writer.writerow([repr(float(x)), repr(float(y))])


def _make_solver_options(backend: str) -> SolverOptions:
    if backend == "mosek_fusion":
        return SolverOptions(backend="mosek_fusion", mosek_params=DEFAULT_MOSEK_PARAMS)
    return SolverOptions(backend="cvxpy")


def _run_layer_scan(
    taus: np.ndarray,
    thetas: np.ndarray,
    h: int,
    alpha: int,
    backend: str,
) -> tuple[list[tuple[float, float]], int]:
    problem = InclusionProblem([Convex(), Convex()])
    algorithm = ChambollePock(tau=float(taus[0]), sigma=float(taus[0]), theta=float(thetas[0]))
    solver_options = _make_solver_options(backend)

    feasible: list[tuple[float, float]] = []
    errors = 0
    processed = 0
    total = len(thetas) * len(taus)

    for row_id, theta in enumerate(thetas, start=1):
        theta_float = float(theta)
        row_feasible = 0
        for tau in taus:
            processed += 1
            tau_float = float(tau)
            algorithm.set_tau(tau_float)
            algorithm.set_sigma(tau_float)
            algorithm.set_theta(theta_float)

            P, p, T, t = IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual(
                algorithm,
                h=h,
                alpha=alpha,
            )
            try:
                result = IterationIndependent.search_lyapunov(
                    problem,
                    algorithm,
                    P,
                    T,
                    p=p,
                    t=t,
                    rho=1.0,
                    h=h,
                    alpha=alpha,
                    solver_options=solver_options,
                    verbosity=0,
                )
            except Exception as exc:
                errors += 1
                print(
                    f"[h={h},a={alpha}] error at tau={tau_float:.6f}, theta={theta_float:.6f}: {exc}"
                )
                continue

            if result.get("status") == "feasible":
                feasible.append((tau_float, theta_float))
                row_feasible += 1

        print(
            f"[h={h},a={alpha}] row {row_id:>2}/{len(thetas)} "
            f"theta={theta_float:.6f} feasible={row_feasible:>3}/{len(taus)} "
            f"cumulative={len(feasible):>5}/{processed:>5} total={total}"
        )

    return feasible, errors


def _scan_layer_worker(
    taus: np.ndarray,
    thetas: np.ndarray,
    h: int,
    alpha: int,
    backend: str,
) -> tuple[int, int, list[tuple[float, float]], int, float]:
    started = time.time()
    points, errors = _run_layer_scan(taus, thetas, h, alpha, backend)
    elapsed = time.time() - started
    return h, alpha, points, errors, elapsed


def _resolve_workers(args: argparse.Namespace) -> int:
    if args.max_workers is not None:
        if args.max_workers < 1:
            raise ValueError("--max-workers must be >= 1.")
        return args.max_workers
    return min(len(LAYER_ORDER), os.cpu_count() or 1)


def _collect_parallel(
    executor: Executor,
    taus: np.ndarray,
    thetas: np.ndarray,
    backend: str,
) -> tuple[dict[tuple[int, int], list[tuple[float, float]]], int]:
    futures: dict[Future, tuple[int, int]] = {}
    for h, alpha in LAYER_ORDER:
        fut = executor.submit(_scan_layer_worker, taus, thetas, h, alpha, backend)
        futures[fut] = (h, alpha)

    points_by_layer: dict[tuple[int, int], list[tuple[float, float]]] = {}
    errors_total = 0
    for fut in as_completed(futures):
        layer = futures[fut]
        try:
            h, alpha, points, errors, elapsed = fut.result()
        except Exception as exc:
            raise RuntimeError(
                f"Layer sweep failed for (h={layer[0]}, alpha={layer[1]}): {exc}"
            ) from exc
        points_by_layer[(h, alpha)] = points
        errors_total += errors
        print(
            f"Layer complete (h={h}, alpha={alpha}): "
            f"feasible={len(points)}, errors={errors}, elapsed={elapsed:.1f}s"
        )
    return points_by_layer, errors_total


def main() -> int:
    args = _build_parser().parse_args()
    _validate(args)

    taus = np.linspace(args.tau_min, args.tau_max, args.tau_count)
    thetas = np.linspace(args.theta_min, args.theta_max, args.theta_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(
        "Grid: "
        f"tau=sigma in [{taus[0]:.6f}, {taus[-1]:.6f}] ({len(taus)} points), "
        f"theta in [{thetas[0]:.6f}, {thetas[-1]:.6f}] ({len(thetas)} points)"
    )
    print("Layers:", ", ".join(f"(h={h}, alpha={a})" for h, a in LAYER_ORDER))
    print()

    started = time.time()
    errors_total = 0

    if args.parallel_layers:
        workers = _resolve_workers(args)
        print(f"Running layer sweeps in parallel with {workers} worker(s)...")
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                points_by_layer, errors_total = _collect_parallel(executor, taus, thetas, args.backend)
        except PermissionError:
            print("Process-based parallelism unavailable; falling back to thread-based parallelism.")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                points_by_layer, errors_total = _collect_parallel(executor, taus, thetas, args.backend)
    else:
        points_by_layer: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for h, alpha in LAYER_ORDER:
            print(f"Running sweep for layer (h={h}, alpha={alpha})...")
            points, errors = _run_layer_scan(taus, thetas, h, alpha, args.backend)
            points_by_layer[(h, alpha)] = points
            errors_total += errors
            print(
                f"Layer complete (h={h}, alpha={alpha}): feasible={len(points)}, errors={errors}"
            )
            print()

    for h, alpha in LAYER_ORDER:
        rows = points_by_layer[(h, alpha)]
        out_path = _output_path(args.output_dir, h, alpha)
        _write_xy(out_path, rows)
        print(f"Wrote {out_path} [{len(rows)} points]")

    elapsed = time.time() - started
    print(f"Finished in {elapsed:.1f}s")
    if errors_total:
        print(f"Solver errors: {errors_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
