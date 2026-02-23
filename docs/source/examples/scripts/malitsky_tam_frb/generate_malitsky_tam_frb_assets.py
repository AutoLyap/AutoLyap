#!/usr/bin/env python3
"""Generate Malitsky-Tam FRB sweep data and an SVG plot asset.

Usage:
    python docs/source/examples/scripts/malitsky_tam_frb/generate_malitsky_tam_frb_assets.py

    # Regenerate only the SVG from an existing CSV table
    python docs/source/examples/scripts/malitsky_tam_frb/generate_malitsky_tam_frb_assets.py --reuse-data
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
from autolyap.algorithms import MalitskyTamFRB
from autolyap.problemclass import (
    InclusionProblem,
    LipschitzOperator,
    MaximallyMonotone,
    StronglyMonotone,
)
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
DATA_REL = Path("data") / "malitsky_tam_frb" / "gamma_rho.csv"
PLOT_IMAGE_REL = Path("_static") / "malitsky_tam_frb_rho_vs_gamma.svg"

# Parameter defaults.
DEFAULT_MU = 1.0
DEFAULT_L = 1.0
GAMMA_MAX = 1.0
GAMMA_POINT_COUNT = 100

# Plot configuration.
PLOT_MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-8,
    "intpntCoTolDfeas": 1e-8,
    "intpntCoTolRelGap": 1e-8,
    "intpntMaxIterations": 1000,
}

PLOT_Y_RANGE = (0.4, 1.0)
PLOT_Y_TICKS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PLOT_WIDTH_PX = 960
PLOT_HEIGHT_PX = PLOT_WIDTH_PX // 2
PLOT_X_LABEL = r"$\gamma$"
PLOT_Y_LABEL = r"$\rho$"
PLOT_Y_LABEL_ROTATION_DEG = 0.0
PLOT_TITLE = ""
PLOT_DESCRIPTION = "Malitsky-Tam FRB rho versus gamma."
PLOT_ARIA_LABEL = "Malitsky-Tam FRB rho versus gamma"
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
        label="Malitsky-Tam (Thm 2.9)",
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
        description="Generate Malitsky-Tam FRB rho-vs-gamma data and SVG plot assets."
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
        "--L", type=float, default=DEFAULT_L, help="Lipschitz parameter L."
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


def _validate_parameters(mu: float, L: float) -> None:
    if not (mu > 0.0):
        raise ValueError(f"Require mu > 0. Got mu={mu}.")
    if not (L > 0.0):
        raise ValueError(f"Require L > 0. Got L={L}.")


def _build_gamma_grid() -> np.ndarray:
    gamma_min = GAMMA_MAX / GAMMA_POINT_COUNT
    return np.linspace(gamma_min, GAMMA_MAX, GAMMA_POINT_COUNT)


def _make_solver_options(_args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(backend="mosek_fusion", mosek_params=PLOT_MOSEK_PARAMS)


def _rho_theory(gamma: float, L: float, mu: float) -> float:
    """Theorem 2.9-derived contraction factor used in the docs comparison."""
    if not (0.0 < gamma < 1.0 / (2.0 * L)):
        raise ValueError("Theorem 2.9 expression requires 0 < gamma < 1/(2L).")
    epsilon = min(0.5 - gamma * L, 5.0 * mu * gamma)
    alpha = min(1.0 + 4.0 * mu * gamma - 0.75 * epsilon, 1.0 + 0.5 * epsilon)
    return 1.0 / alpha


def _build_theory_gamma_grid(L: float) -> np.ndarray:
    gamma_max = 1.0 / (2.0 * L)
    gamma_min = gamma_max * THEORY_GAMMA_MIN_RATIO
    return np.geomspace(gamma_min, gamma_max * (1.0 - THEORY_GAMMA_MIN_RATIO), THEORY_CURVE_POINT_COUNT)


def _run_scan(
    gammas: np.ndarray,
    mu: float,
    L: float,
    solver_options: SolverOptions,
) -> Tuple[List[Tuple[float, float, float]], int]:
    problem = InclusionProblem(
        [
            [MaximallyMonotone(), LipschitzOperator(L=L)],
            StronglyMonotone(mu=mu),
        ]
    )
    algorithm = MalitskyTamFRB(gamma=float(gammas[0]))
    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
    )

    rows: List[Tuple[float, float, float]] = []
    errors = 0

    for row_id, gamma in enumerate(gammas, start=1):
        gamma_float = float(gamma)
        if 0.0 < gamma_float < 1.0 / (2.0 * L):
            rho_theory = _rho_theory(gamma_float, L, mu)
        else:
            rho_theory = float("nan")

        algorithm.set_gamma(gamma_float)

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
            rho_theory_text = f"{rho_theory:.6f}" if np.isfinite(rho_theory) else "nan"
            print(
                f"[scan] {row_id:>3}/{len(gammas)} gamma={gamma_float:>8.6f} "
                f"rho_autolyap={rho_auto_text:>8} rho_theory={rho_theory_text:>8}"
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


def _build_x_ticks() -> Tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.0, GAMMA_MAX, 6))


def _render_plot(
    output_dir: Path,
    rows: Sequence[Tuple[float, float, float]],
    L: float,
    mu: float,
) -> Path:
    theory_points = [
        (float(gamma), _rho_theory(float(gamma), L, mu))
        for gamma in _build_theory_gamma_grid(L)
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
        x_max=GAMMA_MAX,
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
    _validate_parameters(mu=args.mu, L=args.L)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gammas = _build_gamma_grid()
    solver_options = _make_solver_options(args)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    print(f"mu={args.mu}")
    print(f"L={args.L}")
    print(
        f"Gamma grid: {len(gammas)} points on "
        f"(0, {GAMMA_MAX:.6f}]"
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
        print("Running Malitsky-Tam FRB rho sweep...")
        rows, errors = _run_scan(gammas, args.mu, args.L, solver_options)
        data_path = args.output_dir / DATA_REL
        _write_rows(data_path, rows)
        finite_count = sum(1 for _, rho_auto, _ in rows if np.isfinite(rho_auto))
        print(
            f"Sweep complete: finite_rho={finite_count}/{len(rows)}, "
            f"errors={errors}"
        )

    plot_svg_path = _render_plot(args.output_dir, rows, args.L, args.mu)
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
