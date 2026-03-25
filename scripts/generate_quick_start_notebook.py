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

                For theoretical foundations, see [Theory](https://autolyap.github.io/theory.html).
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
        _markdown_cell("## Shared Imports And Helpers"),
        _code_cell(
            dedent(
                """\
                from __future__ import annotations

                import time

                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

                from autolyap import SolverOptions
                from autolyap.algorithms import GradientMethod, OptimizedGradientMethod
                from autolyap.problemclass import (
                    InclusionProblem,
                    SmoothConvex,
                    SmoothStronglyConvex,
                )
                from autolyap.iteration_dependent import IterationDependent
                from autolyap.iteration_independent import IterationIndependent

                AUTOLYAP_BLUE = "#5fa8e8"
                THEORY_BLACK = "#000000"

                plt.style.use("seaborn-v0_8-whitegrid")
                plt.rcParams["figure.dpi"] = 140
                plt.rcParams["axes.spines.top"] = False
                plt.rcParams["axes.spines.right"] = False
                plt.rcParams["axes.facecolor"] = "#f8fafc"
                plt.rcParams["axes.edgecolor"] = "#3f3f46"
                plt.rcParams["axes.linewidth"] = 1.5
                plt.rcParams["grid.color"] = "#d1d5db"
                plt.rcParams["grid.alpha"] = 1.0


                def make_solver_options() -> SolverOptions:
                    return SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")


                def rho_theory(gamma: float, mu: float, L: float) -> float:
                    return max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2


                def run_gradient_sweep(
                    mu: float = 1.0,
                    L: float = 4.0,
                    point_count: int = 100,
                    gamma_start: float | None = None,
                    gamma_end: float | None = None,
                ) -> dict[str, np.ndarray]:
                    if gamma_end is None:
                        gamma_end = 2.0 / L
                    if gamma_start is None:
                        gamma_start = gamma_end / point_count
                    gammas = np.linspace(gamma_start, gamma_end, point_count)

                    solver_options = make_solver_options()
                    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
                    algorithm = GradientMethod(gamma=float(gammas[0]))
                    P, p, T, t = (
                        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                            algorithm
                        )
                    )

                    rows = {
                        "gamma": [],
                        "rho_autolyap": [],
                        "rho_theory": [],
                    }

                    start_time = time.time()
                    for row_id, gamma in enumerate(gammas, start=1):
                        gamma_float = float(gamma)
                        algorithm.set_gamma(gamma_float)
                        rho_theory_value = rho_theory(gamma_float, mu, L)

                        try:
                            result = (
                                IterationIndependent.LinearConvergence.bisection_search_rho(
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
                            )
                        except Exception as exc:
                            rho_autolyap = float("nan")
                            print(f"[gradient sweep] solver error at gamma={gamma_float:.6f}: {exc}")
                        else:
                            if result.get("status") == "feasible":
                                rho_autolyap = float(result["rho"])
                            else:
                                rho_autolyap = float("nan")
                                print(
                                    f"[gradient sweep] no certificate at gamma={gamma_float:.6f}."
                                )

                        rows["gamma"].append(gamma_float)
                        rows["rho_autolyap"].append(rho_autolyap)
                        rows["rho_theory"].append(rho_theory_value)

                        if row_id == 1 or row_id % 10 == 0 or row_id == len(gammas):
                            rho_text = (
                                f"{rho_autolyap:.6f}"
                                if np.isfinite(rho_autolyap)
                                else "nan"
                            )
                            print(
                                f"[gradient sweep] {row_id:>3}/{len(gammas)} "
                                f"gamma={gamma_float:>8.6f} "
                                f"rho_autolyap={rho_text:>8} "
                                f"rho_theory={rho_theory_value:>8.6f}"
                            )

                    elapsed = time.time() - start_time
                    print(f"Gradient sweep completed in {elapsed:.1f}s")
                    return {
                        key: np.asarray(values, dtype=float) for key, values in rows.items()
                    }


                def c_K_theory(algorithm: OptimizedGradientMethod, K: int) -> float:
                    theta = algorithm.compute_theta(K, K)
                    return algorithm.L / (2.0 * theta ** 2)


                def run_optimized_gradient_sweep(
                    L: float = 1.0,
                    k_min: int = 1,
                    k_max: int = 100,
                ) -> dict[str, np.ndarray]:
                    ks = np.arange(k_min, k_max + 1, dtype=int)

                    solver_options = make_solver_options()
                    problem = InclusionProblem([SmoothConvex(L)])

                    rows = {
                        "K": [],
                        "c_K_autolyap": [],
                        "c_K_theory": [],
                    }

                    start_time = time.time()
                    for row_id, K in enumerate(ks, start=1):
                        k_int = int(K)
                        algorithm = OptimizedGradientMethod(L=L, K=k_int)
                        theory_value = c_K_theory(algorithm, k_int)
                        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
                            algorithm,
                            0,
                            i=1,
                            j=1,
                        )
                        Q_K, q_K = (
                            IterationDependent.get_parameters_function_value_suboptimality(
                                algorithm,
                                k_int,
                            )
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
                                verbosity=0,
                            )
                        except Exception as exc:
                            c_K_autolyap = float("nan")
                            print(f"[ogm sweep] solver error at K={k_int}: {exc}")
                        else:
                            if result.get("status") == "feasible":
                                c_K_autolyap = float(result["c_K"])
                            else:
                                c_K_autolyap = float("nan")
                                print(f"[ogm sweep] no certificate at K={k_int}.")

                        rows["K"].append(float(k_int))
                        rows["c_K_autolyap"].append(c_K_autolyap)
                        rows["c_K_theory"].append(theory_value)

                        if row_id == 1 or row_id % 10 == 0 or row_id == len(ks):
                            c_text = (
                                f"{c_K_autolyap:.6e}"
                                if np.isfinite(c_K_autolyap)
                                else "nan"
                            )
                            print(
                                f"[ogm sweep] {row_id:>3}/{len(ks)} "
                                f"K={k_int:>3} "
                                f"c_K_autolyap={c_text:>12} "
                                f"c_K_theory={theory_value:>12.6e}"
                            )

                    elapsed = time.time() - start_time
                    print(f"Optimized-gradient sweep completed in {elapsed:.1f}s")
                    return {
                        key: np.asarray(values, dtype=float) for key, values in rows.items()
                    }


                def plot_gradient_sweep(rows, L: float = 4.0) -> None:
                    fig, ax = plt.subplots(figsize=(12, 4.5))
                    ax.spines["left"].set_color("#3f3f46")
                    ax.spines["bottom"].set_color("#3f3f46")
                    ax.spines["left"].set_linewidth(1.5)
                    ax.spines["bottom"].set_linewidth(1.5)
                    ax.plot(
                        rows["gamma"],
                        rows["rho_theory"],
                        color=THEORY_BLACK,
                        linewidth=2.8,
                        label="Theoretical",
                    )
                    ax.scatter(
                        rows["gamma"],
                        rows["rho_autolyap"],
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


                def plot_optimized_gradient_sweep(rows) -> None:
                    fig, ax = plt.subplots(figsize=(12, 4.5))
                    ax.spines["left"].set_color("#3f3f46")
                    ax.spines["bottom"].set_color("#3f3f46")
                    ax.spines["left"].set_linewidth(1.5)
                    ax.spines["bottom"].set_linewidth(1.5)
                    ax.loglog(
                        rows["K"],
                        rows["c_K_theory"],
                        color=THEORY_BLACK,
                        linewidth=2.8,
                        label="Theoretical",
                    )
                    ax.scatter(
                        rows["K"],
                        rows["c_K_autolyap"],
                        color=AUTOLYAP_BLUE,
                        s=36,
                        alpha=0.9,
                        label="AutoLyap",
                    )
                    ax.set_xlim(0.9, 110.0)
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
                mu = 1.0
                L = 4.0
                gamma = 0.2

                problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
                algorithm = GradientMethod(gamma=gamma)
                solver_options = make_solver_options()

                P, p, T, t = (
                    IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
                        algorithm
                    )
                )

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

                if result["status"] != "feasible":
                    raise RuntimeError(
                        "No feasible Lyapunov certificate in the requested rho interval."
                    )

                rho = result["rho"]
                rho_theory_value = max(abs(1.0 - gamma * L), abs(1.0 - gamma * mu)) ** 2

                print(f"status:       {result['status']}")
                print(f"solve_status: {result['solve_status']}")
                print(f"rho (AutoLyap): {rho:.8f}")
                print(f"rho (theory):   {rho_theory_value:.8f}")
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
                gamma_start = 2.0 / (100.0 * L)
                gamma_end = 2.0 / L

                gradient_sweep = run_gradient_sweep(
                    mu=mu,
                    L=L,
                    point_count=100,
                    gamma_start=gamma_start,
                    gamma_end=gamma_end,
                )
                plot_gradient_sweep(gradient_sweep, L=L)
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
                L = 1.0
                K = 5

                problem = InclusionProblem([SmoothConvex(L)])
                algorithm = OptimizedGradientMethod(L=L, K=K)
                solver_options = make_solver_options()

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

                if result["status"] != "feasible":
                    raise RuntimeError(
                        "No feasible chained Lyapunov certificate for this setup."
                    )

                c_K = result["c_K"]
                theta_K = algorithm.compute_theta(K, K)
                c_K_theory_value = L / (2.0 * theta_K ** 2)

                print(f"status:       {result['status']}")
                print(f"solve_status: {result['solve_status']}")
                print(f"c_K (AutoLyap): {c_K:.6e}")
                print(f"c_K (theory):   {c_K_theory_value:.6e}")
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
                optimized_gradient_sweep = run_optimized_gradient_sweep(
                    L=1.0,
                    k_min=1,
                    k_max=100,
                )
                plot_optimized_gradient_sweep(optimized_gradient_sweep)
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                ## Next
                
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
