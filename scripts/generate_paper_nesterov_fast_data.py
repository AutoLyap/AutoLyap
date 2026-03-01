#!/usr/bin/env python3
"""Generate paper data tables for the Nesterov-fast plot.
Outputs are written under:
  github/paper_data/nesterov_fast_gradient_method/

Files:
  - automatic_lyapunov.tex
  - nesterov_first_bound.tex   (2L/(K+2)^2)
  - nesterov_second_bound.tex  (L/(2 lambda_K^2))
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Allow running from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autolyap import IterationDependent, SolverOptions
from autolyap.algorithms import NesterovFastGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex


def _lambda_k(k: int) -> float:
    lam = 1.0
    for _ in range(k):
        lam = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * lam * lam))
    return lam


def _c_k_nesterov_second(L: float, K: int) -> float:
    """Return L/(2*lambda_K^2)."""
    lam = _lambda_k(K)
    return L / (2.0 * lam * lam)


def _c_k_nesterov_first(L: float, K: int) -> float:
    """Return 2L/(K+2)^2."""
    return 2.0 * L / ((K + 2.0) ** 2)


def _write_xy(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["x", "y"])
        for x, y in rows:
            writer.writerow([x, repr(float(y))])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper_data tables for the Nesterov-fast plot in "
            "overleaf/ver_6/plots/nesterov_fast.tex."
        )
    )
    parser.add_argument("--L", type=float, default=1.0, help="Smoothness constant L > 0.")
    parser.add_argument("--k-min", type=int, default=1, help="Minimum K in sweep (>=1).")
    parser.add_argument("--k-max", type=int, default=100, help="Maximum K in sweep (>=k-min).")
    parser.add_argument(
        "--backend",
        choices=("mosek_fusion", "cvxpy"),
        default="mosek_fusion",
        help="Solver backend passed to SolverOptions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_data" / "nesterov_fast_gradient_method",
        help="Directory where output tables are written.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not (args.L > 0.0):
        raise ValueError(f"Expected L > 0, got {args.L}.")
    if args.k_min < 1:
        raise ValueError(f"Expected k-min >= 1, got {args.k_min}.")
    if args.k_max < args.k_min:
        raise ValueError(f"Expected k-max >= k-min, got {args.k_max} < {args.k_min}.")

    problem = InclusionProblem([SmoothConvex(args.L)])
    algorithm = NesterovFastGradientMethod(gamma=1.0 / args.L)
    solver_options = SolverOptions(backend=args.backend)

    automatic_rows: list[tuple[int, float]] = []
    first_rows: list[tuple[int, float]] = []
    second_rows: list[tuple[int, float]] = []

    for K in range(args.k_min, args.k_max + 1):
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
            algorithm, k=0, i=1, j=2
        )
        Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
            algorithm, k=K, j=2
        )

        result = IterationDependent.search_lyapunov(
            problem,
            algorithm,
            K,
            Q_0,
            Q_K,
            q_0=q_0,
            q_K=q_K,
            solver_options=solver_options,
            verbosity=0,
        )
        if result.get("status") != "feasible":
            raise RuntimeError(
                f"No feasible certificate at K={K}. "
                f"status={result.get('status')}, solve_status={result.get('solve_status')}"
            )

        c_k_autolyap = float(result["c_K"])
        c_k_first = _c_k_nesterov_first(args.L, K)
        c_k_second = _c_k_nesterov_second(args.L, K)

        automatic_rows.append((K, c_k_autolyap))
        first_rows.append((K, c_k_first))
        second_rows.append((K, c_k_second))

        if K == args.k_min or K == args.k_max or (K - args.k_min + 1) % 10 == 0:
            print(
                f"K={K:>3} c_K_auto={c_k_autolyap:.6e} "
                f"c_K_first={c_k_first:.6e} c_K_second={c_k_second:.6e}"
            )

    out_dir = args.output_dir
    automatic_path = out_dir / "automatic_lyapunov.tex"
    first_path = out_dir / "nesterov_first_bound.tex"
    second_path = out_dir / "nesterov_second_bound.tex"

    _write_xy(automatic_path, automatic_rows)
    _write_xy(first_path, first_rows)
    _write_xy(second_path, second_rows)

    print("Wrote:")
    print(f"  - {automatic_path}")
    print(f"  - {first_path}")
    print(f"  - {second_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
