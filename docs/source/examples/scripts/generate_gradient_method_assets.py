#!/usr/bin/env python3
"""Generate gradient-method sweep data and a publication-style SVG plot asset.

Usage:
    python docs/source/examples/scripts/generate_gradient_method_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/generate_gradient_method_assets.py --reuse-data
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
REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import GradientMethod
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex
from plotting_utils import (
    CartesianStyle,
    DEFAULT_SCATTER_COLOR,
    LegendItem,
    LineSeries,
    ScatterSeries,
    render_cartesian_svg,
    write_csv_rows,
)


# Output locations (relative to --output-dir).
DATA_REL = Path("data") / "gradient_method" / "gamma_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "gradient_method_rho_vs_gamma.svg"

# Parameter defaults.
DEFAULT_MU = 1.0
DEFAULT_L = 4.0
GAMMA_POINT_COUNT = 100

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-9,
    "intpntCoTolDfeas": 1e-9,
    "intpntCoTolRelGap": 1e-9,
    "intpntMaxIterations": 700,
}

PLOT_Y_RANGE = (0.3, 1.0)
PLOT_Y_TICKS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$\gamma$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Gradient-method contraction factor vs step size"
PLOT_DESCRIPTION = "Rho vs gamma for gradient method: theoretical curve and AutoLyap points."
PLOT_ARIA_LABEL = "Gradient-method rho versus gamma"
PLOT_SHOW_GRID = True
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
THEORY_COLOR = "#000000"
AUTOLYAP_COLOR = DEFAULT_SCATTER_COLOR
THEORY_LINE_WIDTH_PX = 2.8
AUTOLYAP_MARKER_RADIUS_PX = 4.5
THEORY_CURVE_POINT_COUNT = 1200
THEORY_GAMMA_MIN_RATIO = 1e-4
PLOT_LEGEND = (
    LegendItem(
        label="Theoretical",
        color=THEORY_COLOR,
        kind="line",
        line_width_px=THEORY_LINE_WIDTH_PX,
    ),
    LegendItem(
        label="AutoLyap",
        color=AUTOLYAP_COLOR,
        kind="marker",
        marker_radius_px=AUTOLYAP_MARKER_RADIUS_PX,
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate gradient-method rho-vs-gamma data and SVG plot assets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="Strong-convexity mu.")
    parser.add_argument("--L", type=float, default=DEFAULT_L, help="Smoothness parameter L.")
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


def _validate_parameters(mu: float, L: float) -> None:
    if not (0.0 < mu < L):
        raise ValueError(f"Require 0 < mu < L. Got mu={mu}, L={L}.")


def _build_gamma_grid(L: float) -> np.ndarray:
    gamma_max = 2.0 / L
    gamma_min = gamma_max / GAMMA_POINT_COUNT
    return np.linspace(gamma_min, gamma_max, GAMMA_POINT_COUNT)


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    if args.backend == "cvxpy":
        return SolverOptions(backend="cvxpy", cvxpy_solver=args.cvxpy_solver)
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _rho_theory(gamma: float, mu: float, L: float) -> float:
    return max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2


def _build_theory_gamma_grid(gamma_max: float) -> np.ndarray:
    gamma_min = gamma_max * THEORY_GAMMA_MIN_RATIO
    positive_gammas = np.geomspace(gamma_min, gamma_max, THEORY_CURVE_POINT_COUNT)
    return np.concatenate((np.array([0.0]), positive_gammas))


def _run_scan(
    gammas: np.ndarray,
    mu: float,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=float(gammas[0]))
    P, p, T, t = (
        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            algorithm
        )
    )

    rows: List[Tuple[float, float, float]] = []
    errors = 0

    for row_id, gamma in enumerate(gammas, start=1):
        gamma_float = float(gamma)
        rho_theory = _rho_theory(gamma_float, mu, L)
        algorithm.set_gamma(gamma_float)

        try:
            result = IterationIndependent.LinearConvergence.bisection_search_rho(
                problem,
                algorithm,
                P,
                T,
                p=p,
                t=t,
                S_equals_T=True,
                s_equals_t=True,
                remove_C3=True,
                solver_options=solver_options,
            )
        except Exception as exc:
            errors += 1
            rho_autolyap = float("nan")
            print(f"[scan] solver error at gamma={gamma_float:.6f}: {exc}")
        else:
            if result.get("success", False):
                rho_autolyap = float(result["rho"])
            else:
                errors += 1
                rho_autolyap = float("nan")
                print(f"[scan] no certificate at gamma={gamma_float:.6f}.")

        rows.append((gamma_float, rho_autolyap, rho_theory))
        if row_id == 1 or row_id % 10 == 0 or row_id == len(gammas):
            rho_auto_text = f"{rho_autolyap:.6f}" if np.isfinite(rho_autolyap) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(gammas)} gamma={gamma_float:>8.6f} "
                f"rho_autolyap={rho_auto_text:>8} rho_theory={rho_theory:>8.6f}"
            )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    write_csv_rows(
        path,
        "gamma,rho_autolyap,rho_theory",
        (
            (
                f"{gamma:.12f},"
                f"{rho_autolyap:.12f},"
                f"{rho_theory:.12f}"
            )
            for gamma, rho_autolyap, rho_theory in rows
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
                    float(row["rho_autolyap"]),
                    float(row["rho_theory"]),
                )
            )
    return rows


def _build_x_ticks(L: float) -> Tuple[float, ...]:
    gamma_max = 2.0 / L
    return tuple(float(value) for value in np.linspace(0.0, gamma_max, 6))


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
    mu: float,
    L: float,
) -> Path:
    gamma_max = 2.0 / L
    theory_points = [
        (float(gamma), _rho_theory(float(gamma), mu, L))
        for gamma in _build_theory_gamma_grid(gamma_max)
    ]
    autolyap_points = [
        (gamma, rho_autolyap)
        for gamma, rho_autolyap, _ in rows
        if np.isfinite(rho_autolyap)
    ]
    if not autolyap_points:
        raise RuntimeError("No finite AutoLyap rho values available for plotting.")

    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=0.0,
        x_max=gamma_max,
        y_min=PLOT_Y_RANGE[0],
        y_max=PLOT_Y_RANGE[1],
        x_ticks=_build_x_ticks(L),
        y_ticks=PLOT_Y_TICKS,
        scatter_series=(
            ScatterSeries(
                points=autolyap_points,
                color=AUTOLYAP_COLOR,
                marker_radius_px=AUTOLYAP_MARKER_RADIUS_PX,
            ),
        ),
        line_series=(
            LineSeries(
                points=theory_points,
                color=THEORY_COLOR,
                width_px=THEORY_LINE_WIDTH_PX,
            ),
        ),
        legend_items=PLOT_LEGEND,
        legend_position="bottom-left",
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
    _validate_parameters(mu=args.mu, L=args.L)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gammas = _build_gamma_grid(args.L)
    solver_options = _make_solver_options(args)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    if args.backend == "cvxpy":
        print(f"CVXPY solver: {args.cvxpy_solver}")
    print(f"mu={args.mu}")
    print(f"L={args.L}")
    print(
        f"Gamma grid: {len(gammas)} points on "
        f"(0, 2/L] = (0, {2.0 / args.L:.6f}]"
    )
    print()

    started = time.time()
    if args.reuse_data:
        print("Reusing existing sweep data...")
        rows = _load_rows(args.output_dir)
        errors = 0
        data_path = args.output_dir / DATA_REL
        print(f"Loaded {len(rows)} rows from {data_path}")
    else:
        print("Running gradient-method rho sweep...")
        rows, errors = _run_scan(gammas, args.mu, args.L, solver_options)
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, rho_auto, _ in rows if np.isfinite(rho_auto))
        print(
            f"Sweep complete: finite_rho={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(args.output_dir, rows, args.mu, args.L)
    elapsed = time.time() - started

    print("Finished.")
    print(f"  Data table:  {data_path}")
    print(f"  Plot image:  {plot_svg_path}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    if errors:
        print(f"  Solver errors during sweep: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
