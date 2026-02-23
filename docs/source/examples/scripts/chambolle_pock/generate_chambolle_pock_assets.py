#!/usr/bin/env python3
"""Generate Chambolle--Pock layered sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_assets.py

    # Parallelize layer sweeps across worker processes
    python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_assets.py --parallel-layers

    # Regenerate only the SVG from existing CSV tables
    python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_assets.py --reuse-data
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Allow execution from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = SCRIPTS_ROOT / "shared"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import ChambollePock
from autolyap.problemclass import Convex, InclusionProblem
from plotting_utils import (
    CartesianStyle,
    LegendItem,
    ScatterSeries,
    read_xy_rows,
    render_cartesian_svg,
    write_csv_rows,
)


# Output locations (relative to --output-dir).
DATA_BASE_REL = Path("data") / "chambolle_pock"
DATA_FILES = {
    (0, 0): DATA_BASE_REL / "chambolle_pock_fixed_point_residual_h0_alpha0.csv",
    (1, 0): DATA_BASE_REL / "chambolle_pock_fixed_point_residual_h1_alpha0.csv",
    (1, 1): DATA_BASE_REL / "chambolle_pock_fixed_point_residual_h1_alpha1.csv",
    (2, 0): DATA_BASE_REL / "chambolle_pock_fixed_point_residual_h2_alpha0.csv",
}
PLOT_IMAGE_REL = Path("_static") / "chambolle_pock_fixed_point_residual_layers.svg"

# Sweep grid (dense).
TAU_MIN = 1.0
TAU_MAX = 1.8
TAU_COUNT = 100
THETA_MIN = 0.0
THETA_MAX = 1.5
THETA_COUNT = 100

# Layer definitions for (h, alpha).
LAYER_ORDER = ((2, 0), (1, 1), (1, 0), (0, 0))
LAYER_LABELS = {
    (2, 0): r"$(h,\alpha)=(2,0)$",
    (1, 1): r"$(h,\alpha)=(1,1)$",
    (1, 0): r"$(h,\alpha)=(1,0)$",
    (0, 0): r"$(h,\alpha)=(0,0)$",
}
LAYER_COLORS = {
    (2, 0): "#b56a4a",
    (1, 1): "#0ea5e9",
    (1, 0): "#6f8a6a",
    (0, 0): "#44586d",
}

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-6,
    "intpntCoTolDfeas": 1e-6,
    "intpntCoTolRelGap": 1e-6,
    "intpntMaxIterations": 10000,
}

PLOT_X_RANGE = (0.99, 1.75)
PLOT_Y_RANGE = (0.0, 1.5)
PLOT_X_TICKS = (1.0, 1.15, 1.30, 1.45, 1.60, 1.75)
PLOT_Y_TICKS = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
PLOT_X_LABEL = r"$\tau=\sigma$"
PLOT_Y_LABEL = r"$\theta$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_SHOW_GRID = True
PLOT_WIDTH_PX = 960
# Keep inner plot width and height equal given renderer margins:
# plot_width = width - (96 + 32), plot_height = height - (24 + 88).
PLOT_HEIGHT_PX = 944
PLOT_TITLE = "Chambolle--Pock fixed-point residual summability regions"
PLOT_DESCRIPTION = (
    "Feasible regions in the (tau=sigma, theta) plane for several history/overlap "
    "settings (h, alpha)."
)
PLOT_ARIA_LABEL = "Chambolle-Pock fixed-point residual feasible regions"
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
MARKER_RADIUS_PX = 2.9


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Generate Chambolle--Pock layered sweep data and SVG plot assets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--backend",
        choices=("mosek_fusion",),
        default="mosek_fusion",
        help="AutoLyap backend used for each SDP solve.",
    )
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help=(
            "Skip the expensive sweep and render the SVG from existing "
            "data tables."
        ),
    )
    parser.add_argument(
        "--parallel-layers",
        action="store_true",
        help=(
            "Sweep each (h, alpha) layer in a separate worker process. "
            "Recommended for full regeneration."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum worker processes used with --parallel-layers. "
            "Default: min(number of layers, os.cpu_count())."
        ),
    )
    return parser


def _build_grid() -> Tuple[np.ndarray, np.ndarray]:
    taus = np.linspace(TAU_MIN, TAU_MAX, TAU_COUNT)
    thetas = np.linspace(THETA_MIN, THETA_MAX, THETA_COUNT)
    return taus, thetas


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _run_layer_scan(
    taus: np.ndarray,
    thetas: np.ndarray,
    h: int,
    alpha: int,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float]], int]:
    problem = InclusionProblem([Convex(), Convex()])
    algorithm = ChambollePock(
        tau=float(taus[0]),
        sigma=float(taus[0]),
        theta=float(thetas[0]),
    )

    feasible: List[Tuple[float, float]] = []
    errors = 0
    processed = 0

    for row_id, theta in enumerate(thetas, start=1):
        row_feasible = 0
        for tau in taus:
            processed += 1
            tau_float = float(tau)
            theta_float = float(theta)
            algorithm.set_tau(tau_float)
            algorithm.set_sigma(tau_float)
            algorithm.set_theta(theta_float)

            P, p, T, t = (
                IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual(
                    algorithm,
                    h=h,
                    alpha=alpha,
                )
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
                    f"[scan h={h} a={alpha}] solver error at "
                    f"tau={tau_float:.6f}, theta={theta_float:.6f}: {exc}"
                )
                continue

            if result.get("status") == "feasible":
                feasible.append((tau_float, theta_float))
                row_feasible += 1

        print(
            f"[scan h={h} a={alpha}] row {row_id:>3}/{len(thetas)} "
            f"theta={theta_float:>8.5f} feasible={row_feasible:>3}/{len(taus)} "
            f"cumulative={len(feasible):>5}/{processed}"
        )

    return feasible, errors


def _write_layer_rows(path: Path, rows: Sequence[Tuple[float, float]]) -> None:
    write_csv_rows(
        path,
        "x,y",
        (f"{tau:.12f},{theta:.12f}" for tau, theta in rows),
    )


def _scan_layer_worker(
    taus: np.ndarray,
    thetas: np.ndarray,
    h: int,
    alpha: int,
    mosek_params: Dict[str, float],
) -> Tuple[int, int, List[Tuple[float, float]], int, float]:
    started = time.time()
    solver_options = SolverOptions(backend="mosek_fusion", mosek_params=mosek_params)
    points, errors = _run_layer_scan(taus, thetas, h, alpha, solver_options)
    elapsed = time.time() - started
    return h, alpha, points, errors, elapsed


def _resolve_worker_count(args: argparse.Namespace) -> int:
    if args.max_workers is not None:
        if args.max_workers < 1:
            raise ValueError("--max-workers must be a positive integer.")
        return args.max_workers
    cpu_count = os.cpu_count() or 1
    return min(len(LAYER_ORDER), cpu_count)


def _collect_parallel_layers(
    executor: Executor,
    taus: np.ndarray,
    thetas: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[float, float]]], int]:
    futures: Dict[Future, Tuple[int, int]] = {}
    for h, alpha in LAYER_ORDER:
        future = executor.submit(
            _scan_layer_worker,
            taus,
            thetas,
            h,
            alpha,
            dict(PLOT_MOSEK_PARAMS),
        )
        futures[future] = (h, alpha)

    points_by_layer: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    errors_total = 0
    for future in as_completed(futures):
        layer = futures[future]
        try:
            h, alpha, points, errors, layer_elapsed = future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Layer sweep failed for (h={layer[0]}, alpha={layer[1]}): {exc}"
            ) from exc
        points_by_layer[(h, alpha)] = points
        errors_total += errors
        print(
            f"Layer complete (h={h}, alpha={alpha}): "
            f"feasible={len(points)}, solver_errors={errors}, "
            f"elapsed={layer_elapsed:.1f}s"
        )
        print()

    return points_by_layer, errors_total


def _load_existing_layers(
    output_dir: Path,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    points_by_layer: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    missing = []
    for key in LAYER_ORDER:
        csv_path = output_dir / DATA_FILES[key]
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        points_by_layer[key] = read_xy_rows(csv_path)

    if missing:
        missing_list = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "No sweep data found for one or more layers. Run without --reuse-data first, "
            f"or provide existing tables:{missing_list}"
        )

    return points_by_layer


def _render_layered_plot(
    output_dir: Path,
    points_by_layer: Dict[Tuple[int, int], Sequence[Tuple[float, float]]],
) -> Path:
    # Draw in requested stacking order so later layers appear on top.
    draw_order = ((2, 0), (1, 1), (1, 0), (0, 0))
    scatter_series = tuple(
        ScatterSeries(
            points=points_by_layer[key],
            color=LAYER_COLORS[key],
            marker_radius_px=MARKER_RADIUS_PX,
            opacity=0.90,
        )
        for key in draw_order
    )

    legend_items = tuple(
        LegendItem(
            label=LAYER_LABELS[key],
            color=LAYER_COLORS[key],
            kind="marker",
            marker_radius_px=4.0,
        )
        for key in LAYER_ORDER
    )

    x_min, x_max = PLOT_X_RANGE
    y_min, y_max = PLOT_Y_RANGE
    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=PLOT_X_TICKS,
        y_ticks=PLOT_Y_TICKS,
        scatter_series=scatter_series,
        legend_items=legend_items,
        legend_position="top-right",
        x_label=PLOT_X_LABEL,
        y_label=PLOT_Y_LABEL,
        title=PLOT_TITLE,
        description=PLOT_DESCRIPTION,
        aria_label=PLOT_ARIA_LABEL,
        width_px=PLOT_WIDTH_PX,
        height_px=PLOT_HEIGHT_PX,
        y_label_rotation_deg=PLOT_Y_LABEL_ROTATION_DEG,
        show_grid=PLOT_SHOW_GRID,
        style=PLOT_STYLE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint.

    Parameters:
        argv: Optional argument vector. When omitted, argparse reads from sys.argv.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    taus, thetas = _build_grid()
    solver_options = _make_solver_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(
        f"Grid: tau=sigma in [{taus[0]:.2f}, {taus[-1]:.2f}] ({len(taus)} points), "
        f"theta in [{thetas[0]:.2f}, {thetas[-1]:.2f}] ({len(thetas)} points)"
    )
    print(f"Layers: {', '.join(f'(h={h}, alpha={a})' for h, a in LAYER_ORDER)}")
    if args.parallel_layers and not args.reuse_data:
        workers = _resolve_worker_count(args)
        print(f"Layer execution: parallel ({workers} worker{'s' if workers != 1 else ''})")
    elif not args.reuse_data:
        print("Layer execution: serial")
    print()

    started = time.time()
    if args.reuse_data:
        print("Reusing existing layer data...")
        points_by_layer = _load_existing_layers(args.output_dir)
        errors_total = 0
    else:
        points_by_layer: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        errors_total = 0
        if args.parallel_layers:
            workers = _resolve_worker_count(args)
            print(f"Running layer sweeps in parallel ({workers} worker{'s' if workers != 1 else ''})...")
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    points_by_layer, errors_total = _collect_parallel_layers(
                        executor,
                        taus,
                        thetas,
                    )
            except PermissionError:
                print(
                    "Process-based parallelism unavailable in this environment. "
                    "Falling back to thread-based parallelism."
                )
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    points_by_layer, errors_total = _collect_parallel_layers(
                        executor,
                        taus,
                        thetas,
                    )
        else:
            for h, alpha in LAYER_ORDER:
                print(f"Running sweep for layer (h={h}, alpha={alpha})...")
                points, errors = _run_layer_scan(
                    taus,
                    thetas,
                    h,
                    alpha,
                    solver_options,
                )
                points_by_layer[(h, alpha)] = points
                errors_total += errors
                print(
                    f"Layer complete (h={h}, alpha={alpha}): "
                    f"feasible={len(points)}, solver_errors={errors}"
                )
                print()

        for key in LAYER_ORDER:
            rows = points_by_layer[key]
            data_path = args.output_dir / DATA_FILES[key]
            _write_layer_rows(data_path, rows)

    plot_svg_path = _render_layered_plot(args.output_dir, points_by_layer)

    elapsed = time.time() - started
    print("Finished.")
    for key in LAYER_ORDER:
        data_path = args.output_dir / DATA_FILES[key]
        count = len(points_by_layer[key])
        print(f"  Data (h={key[0]}, alpha={key[1]}): {data_path} [{count} points]")
    print(f"  Plot image: {plot_svg_path}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    if errors_total:
        print(f"  Solver errors during sweeps: {errors_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
