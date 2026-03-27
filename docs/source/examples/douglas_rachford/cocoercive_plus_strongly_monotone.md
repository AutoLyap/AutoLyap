# The Douglas--Rachford method: Cocoercive + strongly monotone

## Problem setup

Consider the monotone inclusion

```{math}
\text{find } x \in \calH \text{ such that } 0 \in G_1(x) + G_2(x),
```

where:

- {math}`G_1 : \calH \to \calH` is {math}`\beta`-cocoercive, with {math}`\beta>0`,
- {math}`G_2 : \calH \rightrightarrows \calH` is {math}`\mu`-strongly monotone
  and maximally monotone, with {math}`\mu>0`.

For an initial point {math}`x^0 \in \calH`, step size
{math}`\gamma \in \reals_{++}`, and relaxation
{math}`\lambda \in \reals`, the Douglas--Rachford update is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
v^k &= J_{\gamma G_1}(x^k), \\
w^k &= J_{\gamma G_2}(2v^k - x^k), \\
x^{k+1} &= x^k + \lambda (w^k - v^k).
\end{aligned}
\right.
```

In this example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \zer(G_1 + G_2).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`G_1` is modeled by
  {py:class}`Cocoercive <autolyap.problemclass.Cocoercive>`.
- {math}`G_2` is modeled by
  {py:class}`StronglyMonotone <autolyap.problemclass.StronglyMonotone>`.
- The full inclusion is built with
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`DouglasRachford <autolyap.algorithms.DouglasRachford>`.
- Distance-to-solution parameters are obtained with
  {py:meth}`IterationIndependent.LinearConvergence.get_parameters_distance_to_solution <autolyap.IterationIndependent.LinearConvergence.get_parameters_distance_to_solution>`.
- The contraction factor is searched with
  {py:meth}`IterationIndependent.LinearConvergence.bisection_search_rho <autolyap.IterationIndependent.LinearConvergence.bisection_search_rho>`.

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
import numpy as np

from autolyap import SolverOptions
from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    Cocoercive,
    InclusionProblem,
    StronglyMonotone,
)
from autolyap.iteration_independent import IterationIndependent

mu = 1.0
beta = 1.0
gamma = 1.0
lambda_value = 1.2

problem = InclusionProblem(
    [
        Cocoercive(beta=beta),  # G1
        StronglyMonotone(mu=mu),  # G2
    ]
)
algorithm = DouglasRachford(gamma=gamma, lambda_value=lambda_value, type="operator")
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

# Build V(P, p, k) and R(T, t, k) for distance-to-solution analysis.
P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
    algorithm
)

result = IterationIndependent.LinearConvergence.bisection_search_rho(
    problem,
    algorithm,
    P,
    T,
    S_equals_T=True,
    s_equals_t=True,
    remove_C3=True,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]

def rho_theoretical(mu: float, beta: float, lambda_value: float, gamma: float) -> float:
    scaled_mu = gamma * mu
    scaled_beta = beta / gamma

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
                - scaled_beta ** 2
                - 2.0 * scaled_mu * scaled_beta ** 2
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
                scaled_mu ** 2
                + scaled_beta ** 2
                + scaled_mu * scaled_beta
                + scaled_mu
                + scaled_beta
                - scaled_mu ** 2 * scaled_beta ** 2
            )
            / (
                scaled_mu ** 2
                + scaled_beta ** 2
                + scaled_mu ** 2 * scaled_beta
                + scaled_mu * scaled_beta ** 2
                + scaled_mu
                + scaled_beta
                - 2.0 * scaled_mu ** 2 * scaled_beta ** 2
            )
        )
    ):
        return abs(
            1.0
            - lambda_value
            * (
                (1.0 + scaled_mu * scaled_beta)
                / ((scaled_mu + 1.0) * (scaled_beta + 1.0))
            )
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
                - scaled_mu ** 2
                - 2.0 * scaled_mu ** 2 * scaled_beta
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


rho_theory = rho_theoretical(mu, beta, lambda_value, gamma)

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

The computed value `rho (AutoLyap)` matches (up to solver numerical tolerances)
the theoretical rate expression in
{cite}`ryu2020operatorsplittingperformance{Theorem 4.1}`, i.e.,

```{math}
\|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{math}
\rho =
\begin{cases}
\left|1-\lambda\dfrac{\tilde{\beta}}{\tilde{\beta}+1}\right|^2,
& \text{if } \tilde{\mu}\tilde{\beta}-\tilde{\mu}+\tilde{\beta}<0
\text{ and } \lambda \le
2\dfrac{(\tilde{\beta}+1)(\tilde{\mu}-\tilde{\beta}-\tilde{\mu}\tilde{\beta})}
{\tilde{\mu}+\tilde{\mu}\tilde{\beta}-\tilde{\beta}-\tilde{\beta}^2-2\tilde{\mu}\tilde{\beta}^2},
\\
\left|1-\lambda\dfrac{1+\tilde{\mu}\tilde{\beta}}{(\tilde{\mu}+1)(\tilde{\beta}+1)}\right|^2,
& \text{if } \tilde{\mu}\tilde{\beta}-\tilde{\mu}-\tilde{\beta}>0
\text{ and } \lambda \le
2\dfrac{\tilde{\mu}^2+\tilde{\beta}^2+\tilde{\mu}\tilde{\beta}+\tilde{\mu}+\tilde{\beta}-\tilde{\mu}^2\tilde{\beta}^2}
{\tilde{\mu}^2+\tilde{\beta}^2+\tilde{\mu}^2\tilde{\beta}+\tilde{\mu}\tilde{\beta}^2+\tilde{\mu}+\tilde{\beta}-2\tilde{\mu}^2\tilde{\beta}^2},
\\
|1-\lambda|^2,
&
\text{if } \lambda \ge
2\dfrac{\tilde{\mu}\tilde{\beta}+\tilde{\mu}+\tilde{\beta}}
{2\tilde{\mu}\tilde{\beta}+\tilde{\mu}+\tilde{\beta}},
\\
\left|1-\lambda\dfrac{\tilde{\mu}}{\tilde{\mu}+1}\right|^2,
& \text{if } \tilde{\mu}\tilde{\beta}+\tilde{\mu}-\tilde{\beta}<0
\text{ and } \lambda \le
2\dfrac{(\tilde{\mu}+1)(\tilde{\beta}-\tilde{\mu}-\tilde{\mu}\tilde{\beta})}
{\tilde{\beta}+\tilde{\mu}\tilde{\beta}-\tilde{\mu}-\tilde{\mu}^2-2\tilde{\mu}^2\tilde{\beta}},
\\
\rho_{5},
& \text{otherwise},
\end{cases}
```

with

```{math}
\begin{aligned}
\rho_{5} &= \dfrac{2-\lambda}{4}
\dfrac{
\big((2-\lambda)\tilde{\mu}(\tilde{\beta}+1)+\lambda\tilde{\beta}(1-\tilde{\mu})\big)
\big((2-\lambda)\tilde{\beta}(\tilde{\mu}+1)+\lambda\tilde{\mu}(1-\tilde{\beta})\big)
}{
\tilde{\mu}\tilde{\beta}\big(2\tilde{\mu}\tilde{\beta}(1-\lambda)+(2-\lambda)(\tilde{\mu}+\tilde{\beta}+1)\big)
},\\
\tilde{\mu} &= \gamma\mu,\\
\tilde{\beta} &= \dfrac{\beta}{\gamma},
\end{aligned}
```

and

```{math}
x^\star \in \zer(G_1 + G_2).
```

Sweeping over 100 values of {math}`\lambda` on {math}`0 < \lambda < 2` (with
{math}`\mu=1`, {math}`\beta=1`, {math}`\gamma=1`) gives the plot below, with the
theoretical rate from {cite}`ryu2020operatorsplittingperformance{Theorem 4.1}`
in black and AutoLyap certificates as blue dots.

```{image} ../../_static/douglas_rachford_cocoercive_plus_strongly_monotone_rho_vs_lambda.svg
:alt: Douglas-Rachford rho versus lambda for cocoercive-plus-strongly-monotone setting with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
