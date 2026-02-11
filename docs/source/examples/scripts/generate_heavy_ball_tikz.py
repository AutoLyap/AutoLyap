#!/usr/bin/env python3
"""Generate heavy-ball sweep data and docs plot assets."""

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
from plotting_utils import render_tex_input_to_png, write_csv_rows


# Output locations (relative to --output-dir).
SMOOTH_DATA_REL = Path("data") / "heavy_ball_smooth_convex" / "gammas_deltas.tex"
PLOT_TEX_REL = Path("plots") / "heavy_ball.tex"
PLOT_IMAGE_REL = Path("_static") / "heavy_ball_smooth_convex.png"

# Single sweep grid (dense).
GAMMA_MIN = 0.05
GAMMA_MAX = 2.60
GAMMA_COUNT = 59
DELTA_MIN = -0.95
DELTA_MAX = 0.95
DELTA_COUNT = 59


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate smooth-convex heavy-ball sweep data and plot assets."
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
    return parser


def _build_grid() -> Tuple[np.ndarray, np.ndarray]:
    gammas = np.linspace(GAMMA_MIN, GAMMA_MAX, GAMMA_COUNT)
    deltas = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_COUNT)
    return gammas, deltas


def _make_solver_options(args: argparse.Namespace) -> SolverOptions:
    if args.backend == "cvxpy":
        return SolverOptions(backend="cvxpy", cvxpy_solver=args.cvxpy_solver)
    return SolverOptions(backend="mosek_fusion")


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
                result = IterationIndependent.verify_iteration_independent_Lyapunov(
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

            if result.get("success", False):
                feasible.append((float(gamma), float(delta)))
                row_feasible += 1

        print(
            f"[smooth] row {row_id:>3}/{len(deltas)} delta={delta:>8.5f} "
            f"feasible={row_feasible:>3}/{len(gammas)} cumulative={len(feasible):>5}/{processed}"
        )

    return feasible, errors


def _write_plot_tex(output_dir: Path) -> Path:
    tex_path = output_dir / PLOT_TEX_REL
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_source = r"""\begin{tikzpicture}
    \begin{axis}[
      width=\textwidth,
      height=\textwidth,
      font=\normalsize,
      label style={font=\normalsize},
      tick label style={font=\normalsize},
      xlabel={\(\gamma\)},
      ylabel={\(\delta\)},
      grid=both,
      xmin=0,
      xmax=2.6,
      ymin=-1,
      ymax=1,
      ylabel style={rotate=-90},
    ]
      \pgfplotstableread[col sep=comma]{data/heavy_ball_smooth_convex/gammas_deltas.tex}\gamdeltatable
      \addplot [
        color={rgb,255:red,95;green,168;blue,232},
        mark options={fill={rgb,255:red,95;green,168;blue,232}, draw={rgb,255:red,95;green,168;blue,232}},
        only marks,
        mark size=1.5pt
      ] table [x=x, y=y] {\gamdeltatable};
    \end{axis}
  \end{tikzpicture}
"""
    tex_path.write_text(tex_source, encoding="utf-8")
    return tex_path


def main(argv: Sequence[str] | None = None) -> int:
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

    plot_tex_path = _write_plot_tex(args.output_dir)
    preview_png = render_tex_input_to_png(
        output_dir=args.output_dir,
        tex_input_relpath=PLOT_TEX_REL,
        png_output_relpath=PLOT_IMAGE_REL,
        wrapper_stem="_heavy_ball_preview_wrapper",
        font_size_pt=12,
        dpi=220,
        preview_border_pt=2,
    )

    elapsed = time.time() - started
    print("Finished.")
    print(f"  Smooth data: {smooth_data}")
    print(f"  Plot source: {plot_tex_path}")
    if preview_png is not None:
        print(f"  Plot image:  {preview_png}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
