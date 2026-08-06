#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Generate triple-momentum smoothness-sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/triple_momentum/generate_triple_momentum_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/triple_momentum/generate_triple_momentum_assets.py --reuse-data
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
from autolyap.algorithms import TripleMomentum
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
DATA_REL = Path("data") / "triple_momentum" / "smoothness_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "triple_momentum_rho_vs_smoothness.svg"

# Parameter defaults.
DEFAULT_MU = 1.0
DEFAULT_L_MAX = 100.0
L_POINT_COUNT = 100

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_Y_RANGE = (0.0, 0.85)
PLOT_Y_TICKS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$L$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Triple-momentum contraction factor vs smoothness parameter"
PLOT_DESCRIPTION = (
    "Rho vs L for triple momentum: theoretical curve and AutoLyap points."
)
PLOT_ARIA_LABEL = "Triple-momentum rho versus smoothness parameter L"
PLOT_SHOW_GRID = True
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
THEORY_COLOR = "#000000"
AUTOLYAP_COLOR = DEFAULT_SCATTER_COLOR
THEORY_LINE_WIDTH_PX = 2.8
AUTOLYAP_MARKER_RADIUS_PX = 4.5
THEORY_CURVE_POINT_COUNT = 1200
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
        description="Generate triple-momentum rho-vs-smoothness data and SVG plot assets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=DEFAULT_MU,
        help="Strong-convexity parameter mu.",
    )
    parser.add_argument(
        "--L-max",
        type=float,
        default=DEFAULT_L_MAX,
        help="Largest smoothness parameter L in the sweep.",
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


def _validate_parameters(mu: float, L_max: float) -> None:
    if not (0.0 < mu < L_max):
        raise ValueError(f"Require 0 < mu < L_max. Got mu={mu}, L_max={L_max}.")


def _build_L_grid(mu: float, L_max: float) -> np.ndarray:
    """Return 100 values on (mu, L_max], explicitly excluding L=mu."""
    return np.linspace(mu, L_max, L_POINT_COUNT + 1)[1:]


def _build_theory_L_grid(mu: float, L_max: float) -> np.ndarray:
    return np.linspace(mu, L_max, THEORY_CURVE_POINT_COUNT)


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _rho_theory(L: float, mu: float) -> float:
    return (1.0 - np.sqrt(mu / L)) ** 2


def _run_scan(
    L_values: np.ndarray,
    mu: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    rows: List[Tuple[float, float, float]] = []
    errors = 0

    for row_id, L in enumerate(L_values, start=1):
        L_float = float(L)
        rho_theory = _rho_theory(L_float, mu)
        problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L_float)])
        algorithm = TripleMomentum(mu=mu, L=L_float)
        P, p, T, t = (
            IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                algorithm
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
                S_equals_T=True,
                s_equals_t=True,
                remove_C3=True,
                solver_options=solver_options,
                verbosity=0,
            )
        except Exception as exc:
            errors += 1
            rho_autolyap = float("nan")
            print(f"[scan] solver error at L={L_float:.6f}: {exc}")
        else:
            if result.get("status") == "feasible":
                rho_autolyap = float(result["rho"])
            else:
                errors += 1
                rho_autolyap = float("nan")
                print(f"[scan] no certificate at L={L_float:.6f}.")

        rows.append((L_float, rho_autolyap, rho_theory))
        if row_id == 1 or row_id % 10 == 0 or row_id == len(L_values):
            rho_auto_text = f"{rho_autolyap:.6f}" if np.isfinite(rho_autolyap) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(L_values)} L={L_float:>10.6f} "
                f"rho_autolyap={rho_auto_text:>8} rho_theory={rho_theory:>8.6f}"
            )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    write_csv_rows(
        path,
        "L,rho_autolyap,rho_theory",
        (
            f"{L:.12f},{rho_autolyap:.12f},{rho_theory:.12f}"
            for L, rho_autolyap, rho_theory in rows
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
                    float(row["L"]),
                    float(row["rho_autolyap"]),
                    float(row["rho_theory"]),
                )
            )
    return rows


def _build_x_ticks(mu: float, L_max: float) -> Tuple[float, ...]:
    ticks = [
        float(value)
        for value in np.linspace(0.0, L_max, 6)
        if value >= mu
    ]
    if not ticks or not np.isclose(ticks[0], mu):
        ticks.insert(0, mu)
    return tuple(ticks)


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
    mu: float,
    L_max: float,
) -> Path:
    theory_points = [
        (float(L), _rho_theory(float(L), mu))
        for L in _build_theory_L_grid(mu, L_max)
    ]
    autolyap_points = [
        (L, rho_autolyap)
        for L, rho_autolyap, _ in rows
        if np.isfinite(rho_autolyap)
    ]
    if not autolyap_points:
        raise RuntimeError("No finite AutoLyap rho values available for plotting.")

    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=mu,
        x_max=L_max,
        y_min=PLOT_Y_RANGE[0],
        y_max=PLOT_Y_RANGE[1],
        x_ticks=_build_x_ticks(mu, L_max),
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
    _validate_parameters(mu=args.mu, L_max=args.L_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    L_values = _build_L_grid(args.mu, args.L_max)
    solver_options = _make_solver_options(args)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}")
    print(f"L_max={args.L_max}")
    print(
        f"L grid: {len(L_values)} points on "
        f"({args.mu:.6f}, {args.L_max:.6f}]"
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
        print("Running triple-momentum smoothness sweep...")
        rows, errors = _run_scan(L_values, args.mu, solver_options)
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, rho_auto, _ in rows if np.isfinite(rho_auto))
        print(
            f"Sweep complete: finite_rho={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(args.output_dir, rows, args.mu, args.L_max)
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
