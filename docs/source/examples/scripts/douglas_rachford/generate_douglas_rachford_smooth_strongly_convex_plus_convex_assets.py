#!/usr/bin/env python3
"""Generate Douglas-Rachford (function) sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_smooth_strongly_convex_plus_convex_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_smooth_strongly_convex_plus_convex_assets.py --reuse-data
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
from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex
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
DATA_REL = (
    Path("data")
    / "douglas_rachford_smooth_strongly_convex_plus_convex"
    / "gamma_rho.csv"
)
PLOT_IMAGE_REL = (
    Path("_static")
    / "douglas_rachford_smooth_strongly_convex_plus_convex_rho_vs_gamma.svg"
)

# Parameter defaults.
DEFAULT_MU = 1.0
DEFAULT_L = 2.0
DEFAULT_LAMBDA = 1.0
GAMMA_MAX = 5.0
GAMMA_POINT_COUNT = 100
BISECTION_TOL = 1e-3

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_Y_RANGE = (0.3, 1.0)
PLOT_Y_TICKS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$\gamma$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Douglas-Rachford contraction factor vs step size"
PLOT_DESCRIPTION = (
    "Rho vs gamma for Douglas-Rachford with smooth strongly-convex plus convex "
    "splitting: theoretical curve and AutoLyap points."
)
PLOT_ARIA_LABEL = "Douglas-Rachford rho versus gamma (smooth strongly-convex plus convex)"
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
    default_output = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Generate Douglas-Rachford (function) rho-vs-gamma data and SVG plot assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu", type=float, default=DEFAULT_MU, help="Strong-convexity parameter mu."
    )
    parser.add_argument(
        "--L", type=float, default=DEFAULT_L, help="Smoothness parameter L."
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Relaxation parameter lambda.",
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
            "Skip the expensive sweep and render the SVG from an existing "
            "data table."
        ),
    )
    return parser


def _validate_parameters(mu: float, L: float, lambda_value: float) -> None:
    if not (mu > 0.0):
        raise ValueError(f"Require mu > 0. Got mu={mu}.")
    if not (L > 0.0):
        raise ValueError(f"Require L > 0. Got L={L}.")
    if not (mu < L):
        raise ValueError(f"Require mu < L. Got mu={mu}, L={L}.")
    if not np.isfinite(lambda_value):
        raise ValueError(f"Require finite lambda_value. Got {lambda_value}.")


def _build_gamma_grid() -> np.ndarray:
    gamma_min = GAMMA_MAX / GAMMA_POINT_COUNT
    return np.linspace(gamma_min, GAMMA_MAX, GAMMA_POINT_COUNT)


def _build_theory_gamma_grid() -> np.ndarray:
    gamma_min = GAMMA_MAX * THEORY_GAMMA_MIN_RATIO
    positive_gammas = np.geomspace(gamma_min, GAMMA_MAX, THEORY_CURVE_POINT_COUNT)
    return np.concatenate((np.array([0.0]), positive_gammas))


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _dr_delta(mu: float, L: float, gamma: float) -> float:
    first = (gamma * L - 1.0) / (gamma * L + 1.0)
    second = (1.0 - gamma * mu) / (1.0 + gamma * mu)
    return float(max(first, second))


def _dr_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    alpha = lambda_value / 2.0
    delta = _dr_delta(mu, L, gamma)
    return (abs(1.0 - alpha) + alpha * delta) ** 2


def _is_admissible(lambda_value: float, mu: float, L: float, gamma: float) -> bool:
    alpha = lambda_value / 2.0
    delta = _dr_delta(mu, L, gamma)
    return (alpha > 0.0) and (alpha < (2.0 / (1.0 + delta)))


def _run_scan(
    gammas: np.ndarray,
    mu: float,
    L: float,
    lambda_value: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L), Convex()])
    algorithm = DouglasRachford(
        gamma=float(gammas[0]),
        lambda_value=lambda_value,
        type="function",
    )
    rows: List[Tuple[float, float, float]] = []
    errors = 0

    for row_id, gamma in enumerate(gammas, start=1):
        gamma_float = float(gamma)
        rho_theory = _dr_rate_sq(lambda_value, mu, L, gamma_float)

        if not _is_admissible(lambda_value, mu, L, gamma_float):
            rho_autolyap = float("nan")
            rows.append((gamma_float, rho_autolyap, rho_theory))
            print(f"[scan] skipped non-admissible gamma={gamma_float:.6f}.")
            continue

        algorithm.set_gamma(gamma_float)
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            algorithm
        )

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
                tol=BISECTION_TOL,
                solver_options=solver_options,
                verbosity=0,
            )
        except Exception as exc:
            errors += 1
            rho_autolyap = float("nan")
            print(f"[scan] solver error at gamma={gamma_float:.6f}: {exc}")
        else:
            if result.get("status") == "feasible":
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
                f"{rho_autolyap:.12e},"
                f"{rho_theory:.12e}"
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


def _build_plot_series(
    rows: Sequence[Tuple[float, float, float]],
    mu: float,
    L: float,
    lambda_value: float,
) -> Tuple[Tuple[LineSeries], Tuple[ScatterSeries]]:
    finite_rows = [row for row in rows if np.isfinite(row[1])]
    auto_points = [(gamma, rho_auto) for gamma, rho_auto, _ in finite_rows]

    gamma_theory = _build_theory_gamma_grid()
    theory_points = [
        (float(gamma), _dr_rate_sq(lambda_value, mu, L, float(gamma)))
        for gamma in gamma_theory
        if _is_admissible(lambda_value, mu, L, float(gamma))
    ]

    line_series = (
        LineSeries(points=theory_points, color=THEORY_COLOR, width_px=THEORY_LINE_WIDTH_PX),
    )
    scatter_series = (
        ScatterSeries(
            points=auto_points,
            color=AUTOLYAP_COLOR,
            marker_radius_px=AUTOLYAP_MARKER_RADIUS_PX,
            opacity=0.92,
        ),
    )
    return line_series, scatter_series


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
    mu: float,
    L: float,
    lambda_value: float,
) -> Path:
    finite_rows = [row for row in rows if np.isfinite(row[1])]
    if not finite_rows:
        raise RuntimeError("No finite AutoLyap rows available to render plot.")

    gammas = np.array([row[0] for row in finite_rows], dtype=float)
    x_min = float(np.min(gammas))
    x_max = float(np.max(gammas))

    y_min, y_max = PLOT_Y_RANGE
    y_ticks = PLOT_Y_TICKS

    line_series, scatter_series = _build_plot_series(rows, mu, L, lambda_value)
    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=tuple(np.linspace(x_min, x_max, 11)),
        y_ticks=y_ticks,
        x_label=PLOT_X_LABEL,
        y_label=PLOT_Y_LABEL,
        line_series=line_series,
        scatter_series=scatter_series,
        legend_items=PLOT_LEGEND,
        width_px=PLOT_WIDTH_PX,
        height_px=PLOT_HEIGHT_PX,
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
    _validate_parameters(args.mu, args.L, args.lambda_value)

    gammas = _build_gamma_grid()
    solver_options = _make_solver_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}")
    print(f"L={args.L}")
    print(f"lambda={args.lambda_value}")
    print(f"Bisection tolerance: {BISECTION_TOL}")
    print(
        f"Grid: gamma in [{gammas[0]:.3f}, {gammas[-1]:.3f}] ({len(gammas)} points)"
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
        print("Running Douglas-Rachford smooth-strongly-convex-plus-convex sweep...")
        rows, errors = _run_scan(
            gammas,
            args.mu,
            args.L,
            args.lambda_value,
            solver_options,
        )
        print(f"Sweep complete: rows={len(rows)}, solver_errors={errors}")
        print()

        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)

    plot_svg_path = _render_plot(
        args.output_dir,
        rows,
        args.mu,
        args.L,
        args.lambda_value,
    )

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
