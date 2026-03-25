#!/usr/bin/env python3
"""Generate the Colab-ready quick-start notebook.

Usage:
    python scripts/generate_quick_start_notebook.py
    python scripts/generate_quick_start_notebook.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "quick_start_colab.ipynb"


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _build_notebook() -> dict[str, object]:
    cells = [
        _markdown_cell(
            dedent(
                """\
                # AutoLyap Quick Start

                This notebook mirrors the two quick-start examples:

                1. Iteration-independent analysis with a bisection search on `rho`.
                2. Iteration-dependent analysis with chained Lyapunov inequalities.
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ## Setup

                Install AutoLyap and Matplotlib. The notebook defaults to
                `backend="cvxpy", cvxpy_solver="CLARABEL"` so it runs without a MOSEK license.
                """
            )
        ),
        _code_cell("%pip install -q autolyap matplotlib"),
        _markdown_cell(
            dedent(
                """\
                ## Iteration-Independent Example: The Gradient Method

                ### Problem Setup

                Consider the unconstrained minimization problem

                $$
                \\underset{x \\in \\mathcal{H}}{\\text{minimize}}\\ f(x),
                $$

                where $f : \\mathcal{H} \\to \\mathbb{R}$ is $\\mu$-strongly convex and $L$-smooth,
                with $0 < \\mu < L$.

                For an initial point $x^0 \\in \\mathcal{H}$ and step size $0 < \\gamma < 2/L$, the
                gradient update is

                $$
                (\\forall k \\in \\mathbb{N})\\quad
                x^{k+1} = x^k - \\gamma \\nabla f(x^k).
                $$

                We use AutoLyap to search for the contraction factor
                $\\rho \\in [0,1)$ with the smallest value among those it certifies such that

                $$
                \\|x^k - x^\\star\\|^2 = O(\\rho^k),
                $$

                where $x^\\star \\in \\operatorname{Argmin}_{x \\in \\mathcal{H}} f(x)$.
                """
            )
        ),
        _code_cell(
            dedent(
                """\
                from autolyap import SolverOptions
                from autolyap.algorithms import GradientMethod
                from autolyap.iteration_independent import IterationIndependent
                from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex


                def run_gradient_method_example(
                    mu=1.0,
                    L=4.0,
                    gamma=0.2,
                ):
                    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
                    algorithm = GradientMethod(gamma=gamma)
                    solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

                    P, p, T, t = (
                        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                            algorithm
                        )
                    )

                    search_result = IterationIndependent.LinearConvergence.bisection_search_rho(
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

                    if search_result["status"] != "feasible":
                        raise RuntimeError(
                            "No feasible Lyapunov certificate in the requested rho interval."
                        )

                    rho_theory = max(abs(1.0 - gamma * L), abs(1.0 - gamma * mu)) ** 2
                    return {
                        "status": search_result["status"],
                        "solve_status": str(search_result["solve_status"]),
                        "rho": float(search_result["rho"]),
                        "rho_theory": rho_theory,
                    }


                mu, L, gamma = 1.0, 4.0, 0.2
                gradient_example = run_gradient_method_example(mu=mu, L=L, gamma=gamma)

                print(f"status:       {gradient_example['status']}")
                print(f"solve_status: {gradient_example['solve_status']}")
                print(f"rho (AutoLyap): {gradient_example['rho']:.8f}")
                print(f"rho (theory):   {gradient_example['rho_theory']:.8f}")
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                The theoretical comparison, from
                [Polyak (1963)](https://doi.org/10.1016/0041-5553(63)90382-3), is

                $$
                \\rho = \\max\\{|1 - \\gamma L|, |1 - \\gamma \\mu|\\}^2,
                $$

                so

                $$
                \\|x^k - x^\\star\\|^2 = O(\\rho^k).
                $$
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ### Plot `rho`-vs-`gamma`

                Run the 100-point docs sweep and compare AutoLyap against the theoretical rate.
                """
            )
        ),
        _code_cell(
            dedent(
                """\
                import time

                import matplotlib.pyplot as plt
                import numpy as np

                AUTOLYAP_BLUE = "#5fa8e8"
                THEORY_BLACK = "#000000"

                plt.style.use("seaborn-v0_8-whitegrid")
                plt.rcParams.update({
                    "figure.dpi": 140,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.facecolor": "#f8fafc",
                    "axes.edgecolor": "#3f3f46",
                    "axes.linewidth": 1.5,
                    "grid.color": "#d1d5db",
                    "grid.alpha": 1.0,
                })

                mu, L = 1.0, 4.0
                gamma_values = np.linspace(2.0 / (100.0 * L), 2.0 / L, 100)
                rho_autolyap_values = []
                rho_theory_values = []

                start_time = time.time()
                for row_id, gamma in enumerate(gamma_values, start=1):
                    gamma_float = float(gamma)
                    try:
                        example = run_gradient_method_example(mu=mu, L=L, gamma=gamma_float)
                        rho_autolyap = float(example["rho"])
                        rho_theory = float(example["rho_theory"])
                    except Exception as exc:
                        rho_autolyap = np.nan
                        rho_theory = max(
                            abs(1.0 - L * gamma_float),
                            abs(1.0 - mu * gamma_float),
                        ) ** 2
                        print(
                            f"[gradient sweep] solver error at gamma={gamma_float:.6f}: "
                            f"{exc}"
                        )

                    rho_autolyap_values.append(rho_autolyap)
                    rho_theory_values.append(rho_theory)

                    if row_id == 1 or row_id % 10 == 0 or row_id == len(gamma_values):
                        rho_text = (
                            f"{rho_autolyap:.6f}"
                            if np.isfinite(rho_autolyap)
                            else "nan"
                        )
                        print(
                            f"[gradient sweep] {row_id:>3}/{len(gamma_values)} "
                            f"gamma={gamma_float:>8.6f} "
                            f"rho_autolyap={rho_text:>8} "
                            f"rho_theory={rho_theory:>8.6f}"
                        )

                elapsed = time.time() - start_time
                print(f"Gradient sweep completed in {elapsed:.1f}s")

                rho_autolyap_values = np.asarray(rho_autolyap_values, dtype=float)
                rho_theory_values = np.asarray(rho_theory_values, dtype=float)

                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.spines["left"].set_color("#3f3f46")
                ax.spines["bottom"].set_color("#3f3f46")
                ax.spines["left"].set_linewidth(1.5)
                ax.spines["bottom"].set_linewidth(1.5)
                ax.plot(
                    gamma_values,
                    rho_theory_values,
                    color=THEORY_BLACK,
                    linewidth=2.8,
                    label="Theoretical",
                )
                ax.scatter(
                    gamma_values,
                    rho_autolyap_values,
                    color=AUTOLYAP_BLUE,
                    s=36,
                    alpha=0.9,
                    label="AutoLyap",
                )
                ax.set_xlim(0.0, 2.0 / L)
                ax.set_ylim(0.3, 1.0)
                ax.set_xlabel(r"$\\gamma$")
                ax.set_ylabel(r"$\\rho$", rotation=0, labelpad=18)
                ax.set_title("Gradient-method contraction factor vs step size")
                ax.legend(
                    frameon=True,
                    facecolor="white",
                    edgecolor="#9ca3af",
                    framealpha=1.0,
                    fancybox=True,
                )
                plt.show()
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ## Iteration-Dependent Example: The Optimized Gradient Method

                ### Problem Setup

                Consider the unconstrained minimization problem

                $$
                \\underset{x \\in \\mathcal{H}}{\\text{minimize}}\\ f(x),
                $$

                where $f : \\mathcal{H} \\to \\mathbb{R}$ is convex and $L$-smooth with $L > 0$.

                For background on the optimized gradient method, see
                [Kim and Fessler (2015)](https://doi.org/10.1007/s10107-015-0949-3).

                For initial points $x^0, y^0 \\in \\mathcal{H}$ and iteration budget
                $K \\in \\mathbb{N}$, the optimized gradient method updates as

                $$
                (\\forall k = 0, \\ldots, K-1)\\quad
                \\left[
                \\begin{aligned}
                    y^{k+1} &= x^k - \\frac{1}{L}\\nabla f(x^k), \\\\
                    x^{k+1} &= y^{k+1}
                    + \\frac{\\theta_k - 1}{\\theta_{k+1}}(y^{k+1} - y^k)
                    + \\frac{\\theta_k}{\\theta_{k+1}}(y^{k+1} - x^k).
                \\end{aligned}
                \\right.
                $$

                $$
                \\theta_k =
                \\begin{cases}
                    1, & \\text{if } k = 0, \\\\
                    \\dfrac{1 + \\sqrt{1 + 4\\theta_{k-1}^2}}{2},
                    & \\text{if } k = 1, \\ldots, K-1, \\\\
                    \\dfrac{1 + \\sqrt{1 + 8\\theta_{k-1}^2}}{2},
                    & \\text{if } k = K.
                \\end{cases}
                $$

                We use AutoLyap to search for the certificate constant
                $c_K$ with the smallest value among those it certifies such that

                $$
                f(x^K) - f(x^\\star) \\le c_K\\|x^0 - x^\\star\\|^2,
                $$

                where $x^\\star \\in \\operatorname{Argmin}_{x \\in \\mathcal{H}} f(x)$.
                """
            )
        ),
        _code_cell(
            dedent(
                """\
                from autolyap import SolverOptions
                from autolyap.algorithms import OptimizedGradientMethod
                from autolyap.iteration_dependent import IterationDependent
                from autolyap.problemclass import InclusionProblem, SmoothConvex


                def run_optimized_gradient_method_example(
                    L=1.0,
                    K=5,
                ):
                    problem = InclusionProblem([SmoothConvex(L)])
                    algorithm = OptimizedGradientMethod(L=L, K=K)
                    solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

                    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
                        algorithm,
                        0,
                        i=1,
                        j=1,
                    )
                    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
                        algorithm,
                        K,
                    )

                    theta_K = algorithm.compute_theta(K, K)
                    c_K_theory = L / (2.0 * theta_K ** 2)

                    search_result = IterationDependent.search_lyapunov(
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

                    if search_result["status"] != "feasible":
                        raise RuntimeError(
                            "No feasible chained Lyapunov certificate for this setup."
                        )

                    return {
                        "status": search_result["status"],
                        "solve_status": str(search_result["solve_status"]),
                        "c_K": float(search_result["c_K"]),
                        "c_K_theory": c_K_theory,
                    }


                L, K = 1.0, 5
                ogm_example = run_optimized_gradient_method_example(L=L, K=K)

                print(f"status:       {ogm_example['status']}")
                print(f"solve_status: {ogm_example['solve_status']}")
                print(f"c_K (AutoLyap): {ogm_example['c_K']:.6e}")
                print(f"c_K (theory):   {ogm_example['c_K_theory']:.6e}")
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                The theoretical comparison is

                $$
                c_K = \\frac{L}{2\\theta_K^2},
                $$

                so

                $$
                f(x^K) - f(x^\\star) = O\\!\\left(\\frac{1}{\\theta_K^2}\\right)
                = O\\!\\left(\\frac{1}{K^2}\\right).
                $$
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ### Plot `c_K`-vs-`K`

                Run the $K = 1, \\ldots, 100$ docs sweep and compare AutoLyap against the
                theoretical bound.
                """
            )
        ),
        _code_cell(
            dedent(
                """\
                import time

                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

                AUTOLYAP_BLUE = "#5fa8e8"
                THEORY_BLACK = "#000000"

                plt.style.use("seaborn-v0_8-whitegrid")
                plt.rcParams.update({
                    "figure.dpi": 140,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.facecolor": "#f8fafc",
                    "axes.edgecolor": "#3f3f46",
                    "axes.linewidth": 1.5,
                    "grid.color": "#d1d5db",
                    "grid.alpha": 1.0,
                })

                L = 1.0
                K_values = np.arange(1, 101, dtype=int)
                c_K_autolyap_values = []
                c_K_theory_values = []

                start_time = time.time()
                for row_id, K in enumerate(K_values, start=1):
                    k_int = int(K)
                    try:
                        example = run_optimized_gradient_method_example(L=L, K=k_int)
                        c_K_autolyap = float(example["c_K"])
                        c_K_theory = float(example["c_K_theory"])
                    except Exception as exc:
                        c_K_autolyap = np.nan
                        theta_K = OptimizedGradientMethod(L=L, K=k_int).compute_theta(k_int, k_int)
                        c_K_theory = L / (2.0 * theta_K ** 2)
                        print(f"[ogm sweep] solver error at K={k_int}: {exc}")

                    c_K_autolyap_values.append(c_K_autolyap)
                    c_K_theory_values.append(c_K_theory)

                    if row_id == 1 or row_id % 10 == 0 or row_id == len(K_values):
                        c_text = (
                            f"{c_K_autolyap:.6e}"
                            if np.isfinite(c_K_autolyap)
                            else "nan"
                        )
                        print(
                            f"[ogm sweep] {row_id:>3}/{len(K_values)} "
                            f"K={k_int:>3} "
                            f"c_K_autolyap={c_text:>12} "
                            f"c_K_theory={c_K_theory:>12.6e}"
                        )

                elapsed = time.time() - start_time
                print(f"Optimized-gradient sweep completed in {elapsed:.1f}s")

                c_K_autolyap_values = np.asarray(c_K_autolyap_values, dtype=float)
                c_K_theory_values = np.asarray(c_K_theory_values, dtype=float)

                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.spines["left"].set_color("#3f3f46")
                ax.spines["bottom"].set_color("#3f3f46")
                ax.spines["left"].set_linewidth(1.5)
                ax.spines["bottom"].set_linewidth(1.5)
                ax.loglog(
                    K_values,
                    c_K_theory_values,
                    color=THEORY_BLACK,
                    linewidth=2.8,
                    label="Theoretical",
                )
                ax.scatter(
                    K_values,
                    c_K_autolyap_values,
                    color=AUTOLYAP_BLUE,
                    s=36,
                    alpha=0.9,
                    label="AutoLyap",
                )
                ax.set_xlim(0.9, float(K_values[-1]) + 10.0)
                ax.set_xlabel(r"$K$")
                ax.set_ylabel(r"$c_{K}$", rotation=0, labelpad=18)
                ax.set_title("Optimized-gradient bound vs iteration budget (log-log)")
                ax.xaxis.set_major_locator(LogLocator(base=10.0))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.grid(which="major", color="#9ca3af", linewidth=1.15)
                ax.grid(which="minor", color="#9ca3af", linewidth=1.15)
                ax.legend(
                    frameon=True,
                    facecolor="white",
                    edgecolor="#9ca3af",
                    framealpha=1.0,
                    fancybox=True,
                )
                plt.show()
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ## Next
                
                - For theoretical foundations, see [Theory](https://autolyap.github.io/theory.html).
                - More worked examples: <https://autolyap.github.io/examples.html>
                - API reference: <https://autolyap.github.io/api_reference.html>
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ## References

                - [Polyak, B. T. (1963)](https://doi.org/10.1016/0041-5553(63)90382-3).
                  *Gradient methods for the minimisation of functionals*. USSR Computational
                  Mathematics and Mathematical Physics, 3(4), 864-878.
                - [Kim, Donghwan and Fessler, Jeffrey A. (2015)](https://doi.org/10.1007/s10107-015-0949-3).
                  *Optimized First-Order Methods for Smooth Convex Minimization*. Mathematical
                  Programming, 159(1-2), 81-107.
                """
            )
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "include_colab_link": False,
                "name": NOTEBOOK_PATH.name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _render_notebook() -> str:
    notebook = _build_notebook()
    return json.dumps(notebook, indent=2, ensure_ascii=False) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Colab-ready AutoLyap quick-start notebook."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if the notebook is not up to date.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rendered = _render_notebook()

    if args.check:
        if not NOTEBOOK_PATH.exists():
            print(f"Notebook is missing: {NOTEBOOK_PATH}", file=sys.stderr)
            return 1
        current = NOTEBOOK_PATH.read_text(encoding="utf-8")
        if current != rendered:
            print(
                "Notebook is out of date. Regenerate it with:\n"
                "  python scripts/generate_quick_start_notebook.py",
                file=sys.stderr,
            )
            return 1
        print(f"Notebook is up to date: {NOTEBOOK_PATH}")
        return 0

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
