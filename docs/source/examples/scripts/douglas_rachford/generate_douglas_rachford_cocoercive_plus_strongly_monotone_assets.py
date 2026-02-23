#!/usr/bin/env python3
"""Generate Douglas-Rachford (operator) lambda-sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_cocoercive_plus_strongly_monotone_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_cocoercive_plus_strongly_monotone_assets.py --reuse-data
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
from autolyap.problemclass import Cocoercive, InclusionProblem, StronglyMonotone
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
DATA_REL = Path("data") / "douglas_rachford_cocoercive_plus_strongly_monotone" / "lambda_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "douglas_rachford_cocoercive_plus_strongly_monotone_rho_vs_lambda.svg"

# Parameter defaults.
DEFAULT_MU = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 1.0
LAMBDA_MIN = 0.01
LAMBDA_MAX = 1.99
LAMBDA_POINT_COUNT = 100

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
PLOT_X_LABEL = r"$\lambda$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = "Douglas-Rachford contraction factor vs relaxation parameter"
PLOT_DESCRIPTION = (
    "Rho vs lambda for Douglas-Rachford: theoretical curve and AutoLyap points."
)
PLOT_ARIA_LABEL = "Douglas-Rachford rho versus lambda"
PLOT_SHOW_GRID = True
PLOT_STYLE = CartesianStyle(grid_color="#9ca3af", grid_width_px=1.35)
THEORY_COLOR = "#000000"
AUTOLYAP_COLOR = DEFAULT_SCATTER_COLOR
THEORY_LINE_WIDTH_PX = 2.8
AUTOLYAP_MARKER_RADIUS_PX = 4.5
THEORY_CURVE_POINT_COUNT = 5000
THEORY_LAMBDA_MIN = 1e-6
THEORY_LAMBDA_MAX = 2.0 - 1e-6
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
            "Generate Douglas-Rachford (operator) rho-vs-lambda data and SVG plot assets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu", type=float, default=DEFAULT_MU, help="Strong-monotonicity parameter mu."
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_BETA, help="Cocoercivity parameter beta."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Fixed step size gamma for the lambda sweep.",
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


def _validate_parameters(mu: float, beta: float, gamma: float) -> None:
    if not (mu > 0.0):
        raise ValueError(f"Require mu > 0. Got mu={mu}.")
    if not (beta > 0.0):
        raise ValueError(f"Require beta > 0. Got beta={beta}.")
    if not (gamma > 0.0):
        raise ValueError(f"Require gamma > 0. Got gamma={gamma}.")


def _build_lambda_grid() -> np.ndarray:
    return np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_POINT_COUNT)


def _build_theory_lambda_grid() -> np.ndarray:
    return np.linspace(THEORY_LAMBDA_MIN, THEORY_LAMBDA_MAX, THEORY_CURVE_POINT_COUNT)


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _rho_theoretical(mu: float, beta: float, lambda_value: float, gamma: float) -> float:
    """Piecewise theoretical rho from Ryu et al., Theorem 4.1."""
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")

    scaled_mu = gamma * mu
    scaled_beta = beta / gamma

    if not (scaled_mu > 0.0 and scaled_beta > 0.0 and 0.0 < lambda_value < 2.0):
        raise ValueError("invalid scaled parameters")

    if (
        (scaled_mu * scaled_beta - scaled_mu + scaled_beta < 0.0)
        and (
            lambda_value
            <= 2.0
            * ((scaled_beta + 1.0) * (scaled_mu - scaled_beta - scaled_mu * scaled_beta))
            / (
                scaled_mu
                + scaled_mu * scaled_beta
                - scaled_beta
                - scaled_beta**2
                - 2.0 * scaled_mu * scaled_beta**2
            )
        )
    ):
        return abs(1.0 - lambda_value * (scaled_beta / (scaled_beta + 1.0))) ** 2

    if (
        (scaled_mu * scaled_beta - scaled_mu - scaled_beta > 0.0)
        and (
            lambda_value
            <= 2.0
            * (
                scaled_mu**2
                + scaled_beta**2
                + scaled_mu * scaled_beta
                + scaled_mu
                + scaled_beta
                - scaled_mu**2 * scaled_beta**2
            )
            / (
                scaled_mu**2
                + scaled_beta**2
                + scaled_mu**2 * scaled_beta
                + scaled_mu * scaled_beta**2
                + scaled_mu
                + scaled_beta
                - 2.0 * scaled_mu**2 * scaled_beta**2
            )
        )
    ):
        return abs(
            1.0
            - lambda_value
            * ((1.0 + scaled_mu * scaled_beta) / ((scaled_mu + 1.0) * (scaled_beta + 1.0)))
        ) ** 2

    if lambda_value >= 2.0 * (scaled_mu * scaled_beta + scaled_mu + scaled_beta) / (
        2.0 * scaled_mu * scaled_beta + scaled_mu + scaled_beta
    ):
        return abs(1.0 - lambda_value) ** 2

    if (
        (scaled_mu * scaled_beta + scaled_mu - scaled_beta < 0.0)
        and (
            lambda_value
            <= 2.0
            * ((scaled_mu + 1.0) * (scaled_beta - scaled_mu - scaled_mu * scaled_beta))
            / (
                scaled_beta
                + scaled_mu * scaled_beta
                - scaled_mu
                - scaled_mu**2
                - 2.0 * scaled_mu**2 * scaled_beta
            )
        )
    ):
        return abs(1.0 - lambda_value * (scaled_mu / (scaled_mu + 1.0))) ** 2

    numerator = (
        (
            (2.0 - lambda_value) * scaled_mu * (scaled_beta + 1.0)
            + lambda_value * scaled_beta * (1.0 - scaled_mu)
        )
        * (
            (2.0 - lambda_value) * scaled_beta * (scaled_mu + 1.0)
            + lambda_value * scaled_mu * (1.0 - scaled_beta)
        )
    )
    denominator = scaled_mu * scaled_beta * (
        2.0 * scaled_mu * scaled_beta * (1.0 - lambda_value)
        + (2.0 - lambda_value) * (scaled_mu + scaled_beta + 1.0)
    )
    return ((np.sqrt(2.0 - lambda_value) / 2.0) * np.sqrt(numerator / denominator)) ** 2


def _run_scan(
    lambda_values: np.ndarray,
    mu: float,
    beta: float,
    gamma: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem([Cocoercive(beta=beta), StronglyMonotone(mu=mu)])
    algorithm = DouglasRachford(
        gamma=gamma,
        lambda_value=float(lambda_values[0]),
        type="operator",
    )
    rows: List[Tuple[float, float, float]] = []
    errors = 0

    for row_id, lambda_value in enumerate(lambda_values, start=1):
        lambda_float = float(lambda_value)

        try:
            rho_theory = _rho_theoretical(mu, beta, lambda_float, gamma)
        except Exception as exc:
            errors += 1
            rho_theory = float("nan")
            print(f"[scan] theory error at lambda={lambda_float:.6f}: {exc}")

        algorithm.set_lambda(lambda_float)
        P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            algorithm
        )

        try:
            result = IterationIndependent.LinearConvergence.bisection_search_rho(
                problem,
                algorithm,
                P,
                T,
                S_equals_T=True,
                s_equals_t=True,
                remove_C3=True,
                solver_options=solver_options,
                verbosity=0,
            )
        except Exception as exc:
            errors += 1
            rho_autolyap = float("nan")
            print(f"[scan] solver error at lambda={lambda_float:.6f}: {exc}")
        else:
            if result.get("status") == "feasible":
                rho_autolyap = float(result["rho"])
            else:
                errors += 1
                rho_autolyap = float("nan")
                print(f"[scan] no certificate at lambda={lambda_float:.6f}.")

        rows.append((lambda_float, rho_autolyap, rho_theory))
        if row_id == 1 or row_id % 10 == 0 or row_id == len(lambda_values):
            rho_auto_text = f"{rho_autolyap:.6f}" if np.isfinite(rho_autolyap) else "nan"
            rho_theory_text = f"{rho_theory:.6f}" if np.isfinite(rho_theory) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(lambda_values)} lambda={lambda_float:>8.6f} "
                f"rho_autolyap={rho_auto_text:>8} rho_theory={rho_theory_text:>8}"
            )

    return rows, errors


def _write_rows(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    write_csv_rows(
        path,
        "lambda,rho_autolyap,rho_theory",
        (
            (
                f"{lambda_value:.12f},"
                f"{rho_autolyap:.12f},"
                f"{rho_theory:.12f}"
            )
            for lambda_value, rho_autolyap, rho_theory in rows
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
                    float(row["lambda"]),
                    float(row["rho_autolyap"]),
                    float(row["rho_theory"]),
                )
            )
    return rows


def _build_x_ticks() -> Tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.0, 2.0, 6))


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
    mu: float,
    beta: float,
    gamma: float,
) -> Path:
    theory_points = [(0.0, 1.0)]
    theory_points.extend(
        [
        (
            float(lambda_value),
            _rho_theoretical(mu, beta, float(lambda_value), gamma),
        )
        for lambda_value in _build_theory_lambda_grid()
        ]
    )
    theory_points.append((2.0, 1.0))
    autolyap_points = [
        (lambda_value, rho_autolyap)
        for lambda_value, rho_autolyap, _ in rows
        if np.isfinite(rho_autolyap)
    ]
    if not autolyap_points:
        raise RuntimeError("No finite AutoLyap rho values available for plotting.")

    return render_cartesian_svg(
        path=output_dir / PLOT_IMAGE_REL,
        x_min=0.0,
        x_max=2.0,
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
                points=theory_points,
                color=THEORY_COLOR,
                width_px=THEORY_LINE_WIDTH_PX,
            ),
        ),
        legend_items=PLOT_LEGEND,
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
    _validate_parameters(mu=args.mu, beta=args.beta, gamma=args.gamma)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambda_values = _build_lambda_grid()
    solver_options = _make_solver_options(args)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}")
    print(f"beta={args.beta}")
    print(f"gamma={args.gamma}")
    print(
        f"Lambda grid: {len(lambda_values)} points on "
        f"[{LAMBDA_MIN:.6f}, {LAMBDA_MAX:.6f}]"
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
        print("Running Douglas-Rachford rho sweep...")
        rows, errors = _run_scan(
            lambda_values,
            args.mu,
            args.beta,
            args.gamma,
            solver_options,
        )
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, rho_auto, _ in rows if np.isfinite(rho_auto))
        print(
            f"Sweep complete: finite_rho={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(
        args.output_dir,
        rows,
        args.mu,
        args.beta,
        args.gamma,
    )
    elapsed = time.time() - started

    print("Finished.")
    print(f"  Data table:  {data_path}")
    print(f"  Plot image:  {plot_svg_path}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    if errors:
        print(f"  Errors during sweep: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
