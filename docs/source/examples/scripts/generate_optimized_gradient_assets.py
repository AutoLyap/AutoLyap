#!/usr/bin/env python3
"""Generate optimized-gradient sweep data and a publication-style SVG plot asset.

Usage:
    python docs/source/examples/scripts/generate_optimized_gradient_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/generate_optimized_gradient_assets.py --reuse-data
"""

from __future__ import annotations

import argparse
import csv
import math
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

from autolyap import IterationDependent, SolverOptions
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex
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
DATA_REL = Path("data") / "optimized_gradient_method" / "k_c.csv"
PLOT_IMAGE_REL = Path("_static") / "optimized_gradient_method_c_vs_K_loglog.svg"

# Parameter defaults.
DEFAULT_L = 1.0
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 100

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-9,
    "intpntCoTolDfeas": 1e-9,
    "intpntCoTolRelGap": 1e-9,
    "intpntMaxIterations": 700,
}

PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$K$"
PLOT_Y_LABEL = r"$c_{K}$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Optimized-gradient bound vs iteration budget (log-log)"
PLOT_DESCRIPTION = (
    "Log-log c_K vs K for optimized gradient method: theoretical curve and AutoLyap points."
)
PLOT_ARIA_LABEL = "Optimized-gradient c_K versus K in log-log scale"
PLOT_SHOW_GRID = True
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
THEORY_COLOR = "#000000"
AUTOLYAP_COLOR = DEFAULT_SCATTER_COLOR
THEORY_LINE_WIDTH_PX = 2.8
AUTOLYAP_MARKER_RADIUS_PX = 4.5
# Optional manual x-range for plotting in K-units.
# Set to `None` to use the data range [min(K), max(K)].
PLOT_X_RANGE_K: Tuple[float, float] | None = (0.9,110)
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
        description="Generate optimized-gradient c_K-vs-K data and SVG plot assets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument("--L", type=float, default=DEFAULT_L, help="Smoothness parameter L.")
    parser.add_argument(
        "--k-min",
        type=int,
        default=DEFAULT_K_MIN,
        help="Minimum iteration budget K in the sweep.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=DEFAULT_K_MAX,
        help="Maximum iteration budget K in the sweep.",
    )
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


def _validate_parameters(L: float, k_min: int, k_max: int) -> None:
    if not (L > 0.0):
        raise ValueError(f"Require L > 0. Got L={L}.")
    if k_min < 1:
        raise ValueError(f"Require k_min >= 1. Got k_min={k_min}.")
    if k_max < k_min:
        raise ValueError(
            f"Require k_max >= k_min. Got k_min={k_min}, k_max={k_max}."
        )


def _build_k_grid(k_min: int, k_max: int) -> np.ndarray:
    return np.arange(k_min, k_max + 1, dtype=int)


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    if args.backend == "cvxpy":
        return SolverOptions(backend="cvxpy", cvxpy_solver=args.cvxpy_solver)
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _c_K_theory(algorithm: OptimizedGradientMethod, K: int) -> float:
    theta = algorithm.compute_theta(K, K)
    return algorithm.L / (2.0 * theta ** 2)


def _run_scan(
    ks: np.ndarray,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[int, float, float]], int]:
    problem = InclusionProblem([SmoothConvex(L)])

    rows: List[Tuple[int, float, float]] = []
    errors = 0

    for row_id, K in enumerate(ks, start=1):
        k_int = int(K)
        algorithm = OptimizedGradientMethod(L=L, K=k_int)
        c_K_theory = _c_K_theory(algorithm, k_int)
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
            algorithm,
            0,
            i=1,
            j=1,
        )
        Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
            algorithm,
            k_int,
        )

        try:
            result = IterationDependent.search_lyapunov(
                problem,
                algorithm,
                k_int,
                Q_0,
                Q_K,
                q_0=q_0,
                q_K=q_K,
                solver_options=solver_options,
            )
        except Exception as exc:
            errors += 1
            c_K_autolyap = float("nan")
            print(f"[scan] solver error at K={k_int}: {exc}")
        else:
            if result.get("status") == "feasible":
                c_K_autolyap = float(result["c_K"])
            else:
                errors += 1
                c_K_autolyap = float("nan")
                print(f"[scan] no certificate at K={k_int}.")

        rows.append((k_int, c_K_autolyap, c_K_theory))
        if row_id == 1 or row_id % 10 == 0 or row_id == len(ks):
            c_K_auto_text = f"{c_K_autolyap:.6e}" if np.isfinite(c_K_autolyap) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(ks)} K={k_int:>3} "
                f"c_K_autolyap={c_K_auto_text:>12} c_K_theory={c_K_theory:>12.6e}"
            )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[int, float, float]]) -> None:
    write_csv_rows(
        path,
        "K,c_K_autolyap,c_K_theory",
        (
            (
                f"{K:d},"
                f"{c_K_autolyap:.12e},"
                f"{c_K_theory:.12e}"
            )
            for K, c_K_autolyap, c_K_theory in rows
        ),
    )


def _load_rows(output_dir: Path) -> List[Tuple[int, float, float]]:
    csv_path = output_dir / DATA_REL
    if not csv_path.exists():
        raise FileNotFoundError(
            "No sweep data found. Run the script without `--reuse-data` first, "
            f"or provide an existing table at:\n  - {csv_path}"
        )

    rows: List[Tuple[int, float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        has_new_schema = {"K", "c_K_autolyap", "c_K_theory"}.issubset(reader.fieldnames)
        has_old_schema = {"K", "c_autolyap", "c_theory"}.issubset(reader.fieldnames)
        if not has_new_schema and not has_old_schema:
            raise ValueError(
                "Unrecognized OGM CSV schema. "
                "Expected headers with either "
                "`K,c_K_autolyap,c_K_theory` or `K,c_autolyap,c_theory`."
            )
        for row in reader:
            if has_new_schema:
                c_k_autolyap = row["c_K_autolyap"]
                c_k_theory = row["c_K_theory"]
            else:
                c_k_autolyap = row["c_autolyap"]
                c_k_theory = row["c_theory"]
            rows.append(
                (
                    int(row["K"]),
                    float(c_k_autolyap),
                    float(c_k_theory),
                )
            )
    return rows


def _power_of_ten_label(exponent: int) -> str:
    return f"10^{{{exponent:d}}}"


def _standard_log_ticks(min_value: float, max_value: float) -> List[Tuple[float, str]]:
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("Log ticks require strictly positive bounds.")
    if max_value <= min_value:
        raise ValueError(
            f"Expected max_value > min_value for log ticks. Got {min_value}, {max_value}."
        )

    min_exp = math.floor(math.log10(min_value))
    max_exp = math.ceil(math.log10(max_value))

    ticks: List[Tuple[float, str]] = []
    for exp in range(min_exp, max_exp + 1):
        decade_base = 10.0 ** exp
        for multiplier in range(1, 10):
            tick_value = multiplier * decade_base
            if tick_value < min_value or tick_value > max_value:
                continue
            tick_label = _power_of_ten_label(exp) if multiplier == 1 else ""
            ticks.append((tick_value, tick_label))

    if ticks:
        return ticks

    # Fallback for ranges narrower than the canonical 1..9 decade grid.
    return [(min_value, f"{min_value:g}"), (max_value, f"{max_value:g}")]


def _log10_pair(x: float, y: float) -> Tuple[float, float]:
    return (math.log10(x), math.log10(y))


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[int, float, float]],
) -> Path:
    theory_points_raw = [(float(K), float(c_K_theory)) for K, _, c_K_theory in rows if c_K_theory > 0.0]
    autolyap_points_raw = [
        (float(K), float(c_K_autolyap))
        for K, c_K_autolyap, _ in rows
        if np.isfinite(c_K_autolyap) and c_K_autolyap > 0.0
    ]
    if not autolyap_points_raw:
        raise RuntimeError("No finite positive AutoLyap c_K values available for plotting.")

    theory_points = [_log10_pair(K, c_K_theory) for K, c_K_theory in theory_points_raw]
    autolyap_points = [_log10_pair(K, c_K_autolyap) for K, c_K_autolyap in autolyap_points_raw]

    all_c_K_values = [c_K for _, c_K in theory_points_raw] + [c_K for _, c_K in autolyap_points_raw]
    c_K_min = min(all_c_K_values)
    c_K_max = max(all_c_K_values)
    c_K_min_exp = math.floor(math.log10(c_K_min))
    c_K_max_exp = math.ceil(math.log10(c_K_max))
    y_min_value = 10.0 ** c_K_min_exp
    y_max_value = 10.0 ** c_K_max_exp
    y_tick_values_with_labels = _standard_log_ticks(y_min_value, y_max_value)
    y_ticks = [math.log10(value) for value, _ in y_tick_values_with_labels]
    y_tick_labels = [label for _, label in y_tick_values_with_labels]

    k_values = [int(K) for K, _, _ in rows]
    k_min = min(k_values)
    k_max = max(k_values)
    if PLOT_X_RANGE_K is None:
        x_min_k = float(k_min)
        x_max_k = float(k_max)
    else:
        x_min_k, x_max_k = PLOT_X_RANGE_K
        if x_min_k <= 0.0:
            raise ValueError(f"PLOT_X_RANGE_K lower bound must be > 0. Got {x_min_k}.")
        if x_max_k <= x_min_k:
            raise ValueError(
                "PLOT_X_RANGE_K upper bound must be greater than lower bound. "
                f"Got {PLOT_X_RANGE_K}."
            )

    x_tick_values_with_labels = _standard_log_ticks(x_min_k, x_max_k)
    x_ticks = [math.log10(value) for value, _ in x_tick_values_with_labels]
    x_tick_labels = [label for _, label in x_tick_values_with_labels]

    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=math.log10(x_min_k),
        x_max=math.log10(x_max_k),
        y_min=math.log10(y_min_value),
        y_max=math.log10(y_max_value),
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_labels=x_tick_labels,
        y_tick_labels=y_tick_labels,
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
        italic_math_labels=True,
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
    _validate_parameters(L=args.L, k_min=args.k_min, k_max=args.k_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ks = _build_k_grid(args.k_min, args.k_max)
    solver_options = _make_solver_options(args)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    if args.backend == "cvxpy":
        print(f"CVXPY solver: {args.cvxpy_solver}")
    print(f"L={args.L}")
    print(
        f"K grid: {len(ks)} points on "
        f"[{int(ks[0])}, {int(ks[-1])}]"
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
        print("Running optimized-gradient c_K sweep...")
        rows, errors = _run_scan(ks, args.L, solver_options)
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, c_K_auto, _ in rows if np.isfinite(c_K_auto))
        print(
            f"Sweep complete: finite_c_K={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(args.output_dir, rows)
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
