#!/usr/bin/env python3
"""Generate heavy-ball sweep data and a publication-style SVG plot asset.

Usage:
    python docs/source/examples/scripts/generate_heavy_ball_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/generate_heavy_ball_assets.py --reuse-data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# Allow execution from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import HeavyBallMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex
from plotting_utils import (
    CartesianStyle,
    read_xy_rows,
    render_scatter_svg,
    write_csv_rows,
)


# Output locations (relative to --output-dir).
SMOOTH_DATA_REL = Path("data") / "heavy_ball_smooth_convex" / "gammas_deltas.csv"
PLOT_IMAGE_REL = Path("_static") / "heavy_ball_smooth_convex.svg"

# Single sweep grid (dense): (min, max, count).
GAMMA_SWEEP = (0.05, 2.61, 100)
DELTA_SWEEP = (-0.99, 0.99, 100)

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-9,
    "intpntCoTolDfeas": 1e-9,
    "intpntCoTolRelGap": 1e-9,
    "intpntMaxIterations": 1000,
}

PLOT_X_RANGE = (0.0, 2.61)
PLOT_Y_RANGE = (-1.0, 1.0)
PLOT_X_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
PLOT_Y_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PLOT_X_LABEL = r"$\gamma$"
PLOT_Y_LABEL = r"$\delta$"
PLOT_Y_LABEL_ROTATION_DEG = 0
PLOT_SHOW_GRID = True
PLOT_TITLE = "Certified heavy-ball smooth-convex region"
PLOT_DESCRIPTION = "Scatter plot of feasible parameter pairs in the (gamma, delta) plane."
PLOT_ARIA_LABEL = "Heavy-ball certified parameter region"
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate smooth-convex heavy-ball sweep data and SVG plot assets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument("--L", type=float, default=1.0, help="Smoothness parameter L.")
    parser.add_argument(
        "--backend",
        choices=("cvxpy", "mosek_fusion"),
        default="mosek_fusion",
        help="AutoLyap backend used for each SDP solve.",
    )
    parser.add_argument(
        "--cvxpy-solver",
        default="CLARABEL",
        help="CVXPY solver name when --backend=cvxpy.",
    )
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help=(
            "Skip the expensive sweep and render the SVG from an existing "
            "data table."
        ),
    )
    return parser


def _build_grid() -> Tuple[np.ndarray, np.ndarray]:
    gamma_min, gamma_max, gamma_count = GAMMA_SWEEP
    delta_min, delta_max, delta_count = DELTA_SWEEP
    gammas = np.linspace(gamma_min, gamma_max, gamma_count)
    deltas = np.linspace(delta_min, delta_max, delta_count)
    return gammas, deltas


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    if args.backend == "cvxpy":
        return SolverOptions(backend="cvxpy", cvxpy_solver=args.cvxpy_solver)
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _run_smooth_convex_scan(
    gammas: np.ndarray,
    deltas: np.ndarray,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float]], int]:
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = HeavyBallMethod(gamma=float(gammas[0]), delta=float(deltas[0]))

    feasible: List[Tuple[float, float]] = []
    errors = 0
    processed = 0

    for row_id, delta in enumerate(deltas, start=1):
        row_feasible = 0
        for gamma in gammas:
            algorithm.set_gamma(float(gamma))
            algorithm.set_delta(float(delta))
            processed += 1

            P, p, T, t = (
                IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality(
                    algorithm,
                    tau=0,
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
                    remove_C4=False,
                    solver_options=solver_options,
                )
            except Exception as exc:
                errors += 1
                print(
                    f"[smooth] solver error at gamma={gamma:.6f}, delta={delta:.6f}: {exc}"
                )
                continue

            if result.get("status") == "feasible":
                feasible.append((float(gamma), float(delta)))
                row_feasible += 1

        print(
            f"[smooth] row {row_id:>3}/{len(deltas)} delta={delta:>8.5f} "
            f"feasible={row_feasible:>3}/{len(gammas)} cumulative={len(feasible):>5}/{processed}"
        )

    return feasible, errors


def _load_existing_points(output_dir: Path) -> List[Tuple[float, float]]:
    csv_path = output_dir / SMOOTH_DATA_REL

    if csv_path.exists():
        return read_xy_rows(csv_path)

    raise FileNotFoundError(
        "No sweep data found. Run the script without `--reuse-data` first, "
        f"or provide an existing table at:\n  - {csv_path}"
    )


def _render_plot(output_dir: Path, smooth_points: Sequence[Tuple[float, float]]) -> Path:
    x_min, x_max = PLOT_X_RANGE
    y_min, y_max = PLOT_Y_RANGE
    return render_scatter_svg(
        path=output_dir / PLOT_IMAGE_REL,
        points=smooth_points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=PLOT_X_TICKS,
        y_ticks=PLOT_Y_TICKS,
        x_label=PLOT_X_LABEL,
        y_label=PLOT_Y_LABEL,
        title=PLOT_TITLE,
        description=PLOT_DESCRIPTION,
        aria_label=PLOT_ARIA_LABEL,
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

    gammas, deltas = _build_grid()
    solver_options = _make_solver_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    if args.backend == "cvxpy":
        print(f"CVXPY solver: {args.cvxpy_solver}")
    print(f"L={args.L}")
    print(
        "Smooth grid: "
        f"gamma in [{gammas[0]:.2f}, {gammas[-1]:.2f}] ({len(gammas)} points), "
        f"delta in [{deltas[0]:.2f}, {deltas[-1]:.2f}] ({len(deltas)} points)"
    )
    print()

    started = time.time()
    if args.reuse_data:
        print("Reusing existing sweep data...")
        smooth_points = _load_existing_points(args.output_dir)
        smooth_errors = 0
        smooth_data = args.output_dir / SMOOTH_DATA_REL
        print(f"Loaded {len(smooth_points)} feasible points from {smooth_data}")
    else:
        print("Running smooth-convex sweep...")
        smooth_points, smooth_errors = _run_smooth_convex_scan(
            gammas, deltas, args.L, solver_options
        )
        print(
            f"Smooth sweep complete: feasible={len(smooth_points)}, solver_errors={smooth_errors}"
        )
        print()

        smooth_data = args.output_dir / SMOOTH_DATA_REL
        write_csv_rows(
            smooth_data,
            "x,y",
            (f"{gamma:.6f},{delta:.6f}" for gamma, delta in smooth_points),
        )

    plot_svg_path = _render_plot(args.output_dir, smooth_points)

    elapsed = time.time() - started
    print("Finished.")
    print(f"  Smooth data: {smooth_data}")
    print(f"  Plot image:  {plot_svg_path}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    if smooth_errors:
        print(f"  Solver errors during sweep: {smooth_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
