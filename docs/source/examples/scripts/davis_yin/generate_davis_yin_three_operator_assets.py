#!/usr/bin/env python3
"""Generate Davis-Yin three-operator sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/davis_yin/generate_davis_yin_three_operator_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/davis_yin/generate_davis_yin_three_operator_assets.py --reuse-data
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
from autolyap.algorithms import DavisYin
from autolyap.problemclass import Convex, InclusionProblem, SmoothConvex, SmoothStronglyConvex
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
DATA_REL = Path("data") / "davis_yin_three_operator" / "l1_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "davis_yin_three_operator_rho_vs_l1.svg"

# Parameter defaults.
DEFAULT_MU_2 = 1.0
DEFAULT_L_2 = 2.0
DEFAULT_LAMBDA_VALUE = 1.0
L1_MAX = 40.0
L1_POINT_COUNT = 100

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_Y_RANGE = (0.2, 1.0)
PLOT_Y_TICKS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$L_1$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Davis-Yin three-operator contraction factor vs smoothness parameter"
PLOT_DESCRIPTION = (
    "Rho vs L1 for Davis-Yin three-operator splitting: two theoretical expressions "
    "and AutoLyap points."
)
PLOT_ARIA_LABEL = "Davis-Yin three-operator rho versus L1"
PLOT_SHOW_GRID = True
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
THEORY_DY_COLOR = "#000000"
THEORY_PG_COLOR = "#6b7280"
THEORY_PG_DASH = "9 7"
AUTOLYAP_COLOR = DEFAULT_SCATTER_COLOR
THEORY_LINE_WIDTH_PX = 2.8
AUTOLYAP_MARKER_RADIUS_PX = 4.5
THEORY_CURVE_POINT_COUNT = 2000
PLOT_LEGEND = (
    LegendItem(
        label="Davis-Yin theoretical rate",
        color=THEORY_DY_COLOR,
        kind="line",
        line_width_px=THEORY_LINE_WIDTH_PX,
    ),
    LegendItem(
        label="Pedregosa-Gidel theoretical rate",
        color=THEORY_PG_COLOR,
        kind="line",
        line_width_px=THEORY_LINE_WIDTH_PX,
        dasharray=THEORY_PG_DASH,
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
            "Generate Davis-Yin three-operator rho-vs-L1 data and SVG plot assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu2",
        type=float,
        default=DEFAULT_MU_2,
        help="Strong-convexity parameter mu2 for f2.",
    )
    parser.add_argument(
        "--L2",
        type=float,
        default=DEFAULT_L_2,
        help="Smoothness parameter L2 for f2.",
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


def _validate_parameters(mu2: float, L2: float) -> None:
    if not (mu2 > 0.0):
        raise ValueError(f"Require mu2 > 0. Got mu2={mu2}.")
    if not (L2 >= mu2):
        raise ValueError(f"Require L2 >= mu2. Got mu2={mu2}, L2={L2}.")


def _build_l1_grid() -> np.ndarray:
    l1_min = L1_MAX / L1_POINT_COUNT
    return np.linspace(l1_min, L1_MAX, L1_POINT_COUNT)


def _build_theory_l1_grid() -> np.ndarray:
    return np.linspace(0.0, L1_MAX, THEORY_CURVE_POINT_COUNT)


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _rho_davis_yin_case6(l1: float, mu2: float, L2: float) -> float:
    """Davis-Yin arXiv version, Theorem D.6 (case 6), with lambda=1."""
    return 1.0 - mu2 / (L2 * (1.0 + l1 / L2) ** 2)


def _rho_pedregosa_gidel(l1: float, mu2: float, L2: float) -> float:
    return 1.0 - min(mu2 / L2, 1.0 / (1.0 + l1 / L2))


def _run_scan(
    l1_values: np.ndarray,
    mu2: float,
    L2: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float, float]], int]:
    gamma = 1.0 / L2
    algorithm = DavisYin(gamma=gamma, lambda_value=DEFAULT_LAMBDA_VALUE)
    P, p, T, t = (
        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            algorithm
        )
    )

    rows: List[Tuple[float, float, float, float]] = []
    errors = 0

    for row_id, l1 in enumerate(l1_values, start=1):
        l1_float = float(l1)
        rho_dy = _rho_davis_yin_case6(l1_float, mu2, L2)
        rho_pg = _rho_pedregosa_gidel(l1_float, mu2, L2)

        problem = InclusionProblem(
            [
                SmoothConvex(L=l1_float),
                SmoothStronglyConvex(mu=mu2, L=L2),
                Convex(),
            ]
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
                solver_options=solver_options,
                verbosity=0,
            )
        except Exception as exc:
            errors += 1
            rho_autolyap = float("nan")
            print(f"[scan] solver error at L1={l1_float:.6f}: {exc}")
        else:
            if result.get("status") == "feasible":
                rho_autolyap = float(result["rho"])
            else:
                errors += 1
                rho_autolyap = float("nan")
                print(f"[scan] no certificate at L1={l1_float:.6f}.")

        rows.append((l1_float, rho_autolyap, rho_dy, rho_pg))
        if row_id == 1 or row_id % 10 == 0 or row_id == len(l1_values):
            rho_auto_text = f"{rho_autolyap:.6f}" if np.isfinite(rho_autolyap) else "nan"
            rho_dy_text = f"{rho_dy:.6f}" if np.isfinite(rho_dy) else "nan"
            rho_pg_text = f"{rho_pg:.6f}" if np.isfinite(rho_pg) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(l1_values)} L1={l1_float:>9.6f} "
                f"rho_autolyap={rho_auto_text:>8} rho_dy={rho_dy_text:>8} "
                f"rho_pg={rho_pg_text:>8}"
            )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float, float]]) -> None:
    write_csv_rows(
        path,
        "l1,rho_autolyap,rho_davis_yin,rho_pedregosa_gidel",
        (
            (
                f"{l1:.12f},"
                f"{rho_autolyap:.12f},"
                f"{rho_dy:.12f},"
                f"{rho_pg:.12f}"
            )
            for l1, rho_autolyap, rho_dy, rho_pg in rows
        ),
    )


def _load_rows(output_dir: Path) -> List[Tuple[float, float, float, float]]:
    csv_path = output_dir / DATA_REL
    if not csv_path.exists():
        raise FileNotFoundError(
            "No sweep data found. Run the script without `--reuse-data` first, "
            f"or provide an existing table at:\n  - {csv_path}"
        )

    rows: List[Tuple[float, float, float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["l1"]),
                    float(row["rho_autolyap"]),
                    float(row["rho_davis_yin"]),
                    float(row["rho_pedregosa_gidel"]),
                )
            )
    return rows


def _build_x_ticks() -> Tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.0, L1_MAX, 6))


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float, float]],
    mu2: float,
    L2: float,
) -> Path:
    theory_l1_values = _build_theory_l1_grid()
    theory_dy_points = [
        (float(l1), _rho_davis_yin_case6(float(l1), mu2, L2)) for l1 in theory_l1_values
    ]
    theory_pg_points = [
        (float(l1), _rho_pedregosa_gidel(float(l1), mu2, L2)) for l1 in theory_l1_values
    ]
    autolyap_points = [
        (l1, rho_autolyap)
        for l1, rho_autolyap, _, _ in rows
        if np.isfinite(rho_autolyap)
    ]
    if not autolyap_points:
        raise RuntimeError("No finite AutoLyap rho values available for plotting.")

    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=0.0,
        x_max=L1_MAX,
        y_min=PLOT_Y_RANGE[0],
        y_max=PLOT_Y_RANGE[1],
        x_ticks=_build_x_ticks(),
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
                points=theory_dy_points,
                color=THEORY_DY_COLOR,
                width_px=THEORY_LINE_WIDTH_PX,
            ),
            LineSeries(
                points=theory_pg_points,
                color=THEORY_PG_COLOR,
                width_px=THEORY_LINE_WIDTH_PX,
                dasharray=THEORY_PG_DASH,
            ),
        ),
        legend_items=PLOT_LEGEND,
        legend_position="bottom-right",
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
    _validate_parameters(mu2=args.mu2, L2=args.L2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    l1_values = _build_l1_grid()
    solver_options = _make_solver_options(args)

    gamma = 1.0 / args.L2
    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu2={args.mu2}")
    print(f"L2={args.L2}")
    print(f"lambda={DEFAULT_LAMBDA_VALUE}")
    print(f"gamma=1/L2={gamma:.12f}")
    print(
        f"L1 grid: {len(l1_values)} points on "
        f"(0, {L1_MAX:.6f}]"
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
        print("Running Davis-Yin three-operator rho sweep...")
        rows, errors = _run_scan(l1_values, args.mu2, args.L2, solver_options)
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, rho_auto, _, _ in rows if np.isfinite(rho_auto))
        print(
            f"Sweep complete: finite_rho={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(args.output_dir, rows, args.mu2, args.L2)
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
