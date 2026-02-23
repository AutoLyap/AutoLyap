#!/usr/bin/env python3
"""Generate constant-Nesterov-momentum (gradient-dominated + smooth) assets.

Usage:
    python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_gradient_dominated_smooth_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_gradient_dominated_smooth_assets.py --reuse-data
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
from autolyap.algorithms import GradientNesterovMomentum
from autolyap.problemclass import GradientDominated, InclusionProblem, Smooth
from plotting_utils import CartesianStyle, render_heat_scatter_svg, write_csv_rows


# Output locations (relative to --output-dir).
DATA_REL = (
    Path("data")
    / "nesterov_momentum_gradient_dominated_smooth"
    / "gammas_deltas_rho.csv"
)
PLOT_IMAGE_REL = Path("_static") / "nesterov_momentum_gradient_dominated_smooth.svg"

# Parameter defaults.
DEFAULT_MU_GD = 0.5
DEFAULT_L = 1.0

# Sweep grid: (min, max, count).
GAMMA_SWEEP = (0.01, 3.16, 100)
DELTA_SWEEP = (-1, 1, 100)

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_X_RANGE = (0.0, 3.3)
PLOT_Y_RANGE = (-1.0, 1.0)
PLOT_X_TICKS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3)
PLOT_Y_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PLOT_COLOR_RANGE = (0.4, 1.0)
PLOT_COLOR_TICKS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PLOT_X_LABEL = r"$\gamma$"
PLOT_Y_LABEL = r"$\delta$"
PLOT_COLOR_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_SHOW_GRID = True
PLOT_TITLE = (
    "Certified constant-Nesterov-momentum linear rates "
    "(gradient-dominated + smooth)"
)
PLOT_DESCRIPTION = (
    "Rho over feasible constant-Nesterov-momentum (gamma, delta) pairs for "
    "gradient-dominated, smooth objectives."
)
PLOT_ARIA_LABEL = "Constant-Nesterov-momentum rho over gamma and delta"
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Generate constant-Nesterov-momentum (gradient-dominated + smooth) "
            "sweep data and SVG plot assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu-gd",
        type=float,
        default=DEFAULT_MU_GD,
        help="Gradient-dominance parameter mu_gd.",
    )
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


def _validate_parameters(mu_gd: float, L: float) -> None:
    if not (mu_gd > 0.0):
        raise ValueError(f"Require mu_gd > 0. Got mu_gd={mu_gd}.")
    if not (L > 0.0):
        raise ValueError(f"Require L > 0. Got L={L}.")


def _build_grid() -> Tuple[np.ndarray, np.ndarray]:
    gamma_min, gamma_max, gamma_count = GAMMA_SWEEP
    delta_min, delta_max, delta_count = DELTA_SWEEP
    gammas = np.linspace(gamma_min, gamma_max, gamma_count)
    deltas = np.linspace(delta_min, delta_max, delta_count)
    return gammas, deltas


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _run_scan(
    gammas: np.ndarray,
    deltas: np.ndarray,
    mu_gd: float,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem(
        [
            [GradientDominated(mu_gd=mu_gd), Smooth(L=L)],
        ]
    )
    algorithm = GradientNesterovMomentum(
        gamma=float(gammas[0]),
        delta=float(deltas[0]),
    )

    rows: List[Tuple[float, float, float]] = []
    errors = 0
    processed = 0

    for row_id, delta in enumerate(deltas, start=1):
        row_feasible = 0

        for gamma in gammas:
            processed += 1
            algorithm.set_gamma(float(gamma))
            algorithm.set_delta(float(delta))
            P, p, T, t = (
                IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality(
                    algorithm,
                    tau=0,
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
                    solver_options=solver_options,
                    verbosity=0,
                )
            except Exception as exc:
                errors += 1
                print(
                    f"[scan] solver error at gamma={gamma:.6f}, delta={delta:.6f}: {exc}"
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

            rows.append((float(gamma), float(delta), rho_float))
            row_feasible += 1

        print(
            f"[scan] row {row_id:>3}/{len(deltas)} delta={delta:>8.5f} "
            f"feasible={row_feasible:>3}/{len(gammas)} cumulative={len(rows):>5}/{processed}"
        )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    write_csv_rows(
        path,
        "gamma,delta,rho",
        (
            (
                f"{gamma:.12f},"
                f"{delta:.12f},"
                f"{rho:.12f}"
            )
            for gamma, delta, rho in rows
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
                    float(row["gamma"]),
                    float(row["delta"]),
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
    _validate_parameters(args.mu_gd, args.L)

    gammas, deltas = _build_grid()
    solver_options = _make_solver_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu_gd={args.mu_gd}")
    print(f"L={args.L}")
    print(
        f"Grid: gamma in [{gammas[0]:.2f}, {gammas[-1]:.2f}] ({len(gammas)} points), "
        f"delta in [{deltas[0]:.2f}, {deltas[-1]:.2f}] ({len(deltas)} points)"
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
        print("Running constant-Nesterov-momentum gradient-dominated sweep...")
        rows, errors = _run_scan(
            gammas,
            deltas,
            args.mu_gd,
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
