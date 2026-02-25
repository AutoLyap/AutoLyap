#!/usr/bin/env python3
"""Generate Chambolle--Pock (smooth + strongly convex) sweep data and an SVG plot.

Usage:
    python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_smooth_strongly_convex_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_smooth_strongly_convex_assets.py --reuse-data
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

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
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex
from plotting_utils import CartesianStyle, render_heat_scatter_svg, write_csv_rows


# Output locations (relative to --output-dir).
DATA_REL = Path("data") / "chambolle_pock_smooth_strongly_convex" / "tau_theta_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "chambolle_pock_smooth_strongly_convex_rho.svg"

# Parameter defaults (matching the Figure 4b setting in
# Upadhyaya--Taylor--Drori, 2025, with tau=sigma >= 0.5).
DEFAULT_MU = 0.05
DEFAULT_L = 50.0
BISECTION_TOL = 1e-3

# Dense sweep grid: (min, max, count).
TAU_SWEEP = (0.50, 1.75, 100)
THETA_SWEEP = (0.00, 8.00, 100)

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_X_RANGE = (0.50, 1.75)
PLOT_Y_RANGE = (0.00, 8.00)
PLOT_X_TICKS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75)
PLOT_Y_TICKS = (0.0, 2.0, 4.0, 6.0, 8.0)
PLOT_COLOR_RANGE = (0.88, 1.00)
PLOT_COLOR_TICKS = (0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00)
PLOT_X_LABEL = r"$\tau=\sigma$"
PLOT_Y_LABEL = r"$\theta$"
PLOT_COLOR_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_SHOW_GRID = True
PLOT_TITLE = (
    "Certified Chambolle--Pock linear rates "
    "(smooth + strongly convex)"
)
PLOT_DESCRIPTION = (
    "Rho over feasible Chambolle-Pock (tau=sigma, theta) pairs for smooth, "
    "strongly convex objectives."
)
PLOT_ARIA_LABEL = "Chambolle-Pock rho over tau equals sigma and theta"
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chambolle--Pock (smooth + strongly convex) sweep data and SVG plot assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="Strong-convexity parameter mu.")
    parser.add_argument("--L", type=float, default=DEFAULT_L, help="Smoothness parameter L.")
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
            "Skip the expensive sweep and render the SVG from an existing "
            "data table."
        ),
    )
    return parser


def _validate_parameters(mu: float, L: float) -> None:
    if not (0.0 < mu < L):
        raise ValueError(f"Require 0 < mu < L. Got mu={mu}, L={L}.")


def _build_grid() -> Tuple[np.ndarray, np.ndarray]:
    tau_min, tau_max, tau_count = TAU_SWEEP
    theta_min, theta_max, theta_count = THETA_SWEEP
    taus = np.linspace(tau_min, tau_max, tau_count)
    thetas = np.linspace(theta_min, theta_max, theta_count)
    return taus, thetas


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _run_scan(
    taus: np.ndarray,
    thetas: np.ndarray,
    mu: float,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem(
        [
            SmoothStronglyConvex(mu=mu, L=L),
            SmoothStronglyConvex(mu=mu, L=L),
        ]
    )
    algorithm = ChambollePock(
        tau=float(taus[0]),
        sigma=float(taus[0]),
        theta=float(thetas[0]),
    )

    rows: List[Tuple[float, float, float]] = []
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
                IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                    algorithm,
                    i=1,
                )
            )

            try:
                result = IterationIndependent.LinearConvergence.bisection_search_rho(
                    problem,
                    algorithm,
                    P,
                    T,
                    p=p,
                    t=t,
                    tol=BISECTION_TOL,
                    solver_options=solver_options,
                    verbosity=0,
                )
            except Exception as exc:
                errors += 1
                print(
                    f"[scan] solver error at tau=sigma={tau_float:.6f}, "
                    f"theta={theta_float:.6f}: {exc}"
                )
                continue

            if result.get("status") != "feasible":
                continue

            rho_value = result.get("rho")
            if rho_value is None:
                continue
            rho_float = float(rho_value)
            if not np.isfinite(rho_float):
                continue

            rows.append((tau_float, theta_float, rho_float))
            row_feasible += 1

        print(
            f"[scan] row {row_id:>3}/{len(thetas)} theta={theta:>8.5f} "
            f"feasible={row_feasible:>3}/{len(taus)} cumulative={len(rows):>5}/{processed}"
        )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    write_csv_rows(
        path,
        "tau,theta,rho",
        (
            (
                f"{tau:.12f},"
                f"{theta:.12f},"
                f"{rho:.12f}"
            )
            for tau, theta, rho in rows
        ),
    )


def _load_rows(output_dir: Path) -> List[Tuple[float, float, float]]:
    csv_path = output_dir / DATA_REL
    if not csv_path.exists():
        raise FileNotFoundError(
            "No sweep data found. Run the script without `--reuse-data` first, "
            f"or provide an existing table at:\n  - {csv_path}"
        )

    rows: List[Tuple[float, float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["tau"]),
                    float(row["theta"]),
                    float(row["rho"]),
                )
            )
    return rows


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
) -> Path:
    x_min, x_max = PLOT_X_RANGE
    y_min, y_max = PLOT_Y_RANGE
    color_min, color_max = PLOT_COLOR_RANGE
    return render_heat_scatter_svg(
        path=output_dir / PLOT_IMAGE_REL,
        points=rows,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=PLOT_X_TICKS,
        y_ticks=PLOT_Y_TICKS,
        x_label=PLOT_X_LABEL,
        y_label=PLOT_Y_LABEL,
        colorbar_label=PLOT_COLOR_LABEL,
        value_min=color_min,
        value_max=color_max,
        value_ticks=PLOT_COLOR_TICKS,
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
    _validate_parameters(args.mu, args.L)

    taus, thetas = _build_grid()
    solver_options = _make_solver_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}")
    print(f"L={args.L}")
    print(f"Bisection tolerance: {BISECTION_TOL}")
    print(
        f"Grid: tau=sigma in [{taus[0]:.2f}, {taus[-1]:.2f}] ({len(taus)} points), "
        f"theta in [{thetas[0]:.2f}, {thetas[-1]:.2f}] ({len(thetas)} points)"
    )
    print()

    started = time.time()
    if args.reuse_data:
        print("Reusing existing sweep data...")
        rows = _load_rows(args.output_dir)
        errors = 0
        data_path = args.output_dir / DATA_REL
        print(f"Loaded {len(rows)} feasible points from {data_path}")
    else:
        print("Running Chambolle--Pock smooth-strongly-convex sweep...")
        rows, errors = _run_scan(
            taus,
            thetas,
            args.mu,
            args.L,
            solver_options,
        )
        print(
            f"Sweep complete: feasible={len(rows)}, solver_errors={errors}"
        )
        print()

        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)

    plot_svg_path = _render_plot(args.output_dir, rows)

    elapsed = time.time() - started
    print("Finished.")
    print(f"  Data:       {data_path}")
    print(f"  Plot image: {plot_svg_path}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    if errors:
        print(f"  Solver errors during sweep: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
