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

                Install AutoLyap. The notebook defaults to
                `backend="cvxpy", cvxpy_solver="CLARABEL"` so it runs without a MOSEK license.
                """
            )
        ),
        _code_cell("%pip install -q autolyap"),
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
                \\|x^k - x^\\star\\|^2 \\in \\mathcal{O}(\\rho^k) \\quad \\textup{ as } \\quad k\\to\\infty,
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


                mu, L, gamma = 1.0, 4.0, 0.2
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

                print(f"status:       {search_result['status']}")
                print(f"solve_status: {search_result['solve_status']}")
                print(f"rho (AutoLyap): {float(search_result['rho']):.8f}")
                print(f"rho (theory):   {rho_theory:.8f}")
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                The theoretical comparison, from
                [Polyak (1963)](https://doi.org/10.1016/0041-5553(63)90382-3), is

                $$
                \\|x^k - x^\\star\\|^2 \\in \\mathcal{O}(\\rho^k) \\quad \\textup{ as } \\quad k\\to\\infty, \\qquad
                \\rho = \\max\\{|1-\\gamma L|,\\;|1-\\gamma\\mu|\\}^2,
                $$

                where $x^\\star \\in \\operatorname{Argmin}_{x \\in \\mathcal{H}} f(x)$.

                Equivalently,

                $$
                \\|x^k - x^\\star\\| \\in \\mathcal{O}\\!\\left(\\max\\{|1-\\gamma L|,\\;|1-\\gamma\\mu|\\}^k\\right)
                \\quad \\textup{ as } \\quad k\\to\\infty.
                $$
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
                (\\forall k \\in \\llbracket 0, K-1 \\rrbracket)\\quad
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
                    & \\text{if } k \\in \\llbracket 1, K-1 \\rrbracket, \\\\
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


                L, K = 1.0, 5
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

                print(f"status:       {search_result['status']}")
                print(f"solve_status: {search_result['solve_status']}")
                print(f"c_K (AutoLyap): {float(search_result['c_K']):.6e}")
                print(f"c_K (theory):   {c_K_theory:.6e}")
                """
            )
        ),
        _markdown_cell(
            dedent(
                """\
                The theoretical comparison is

                $$
                f(x^K) - f(x^\\star) \\le c_K\\,\\|x^0 - x^\\star\\|^2, \\qquad
                c_K = \\frac{L}{2\\theta_K^2}.
                $$

                where $x^\\star \\in \\operatorname{Argmin}_{x \\in \\mathcal{H}} f(x)$.

                In particular,

                $$
                f(x^K) - f(x^\\star) \\in \\mathcal{O}\\!\\left(\\frac{1}{\\theta_K^2}\\right)
                \\quad \\textup{ as } \\quad K\\to\\infty,
                \\qquad
                f(x^K) - f(x^\\star) \\in \\mathcal{O}\\!\\left(\\frac{1}{K^2}\\right)
                \\quad \\textup{ as } \\quad K\\to\\infty.
                $$
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
