# Douglas--Rachford: Maximally monotone/Lipschitz + strongly monotone

## Problem setup

Consider the monotone inclusion

```{math}
\text{find } x \in \calH \text{ such that } 0 \in G_1(x) + G_2(x),
```

where:

- {math}`G_1 : \calH \to \calH` is monotone and
  {math}`L`-Lipschitz continuous, with {math}`L>0`,
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
   \|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \zer(G_1 + G_2).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`G_1` is modeled as an intersection of
  {py:class}`MaximallyMonotone <autolyap.problemclass.MaximallyMonotone>` and
  {py:class}`LipschitzOperator <autolyap.problemclass.LipschitzOperator>`.
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
    InclusionProblem,
    LipschitzOperator,
    MaximallyMonotone,
    StronglyMonotone,
)
from autolyap.iteration_independent import IterationIndependent

mu = 1.0
L = 1.0
gamma = 1.0
lambda_value = 1.2

problem = InclusionProblem(
    [
        [MaximallyMonotone(), LipschitzOperator(L=L)],  # G1
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

def rho_theoretical(mu: float, L: float, lambda_value: float, gamma: float) -> float:
    scaled_mu = gamma * mu
    scaled_L = gamma * L

    term1 = 2.0 * (lambda_value - 1.0) * scaled_mu + lambda_value - 2.0
    term2 = scaled_L ** 2 * (lambda_value - 2.0 * (scaled_mu + 1.0))
    denominator = np.sqrt(term1 ** 2 + scaled_L ** 2 * (lambda_value - 2.0 * (scaled_mu + 1.0)) ** 2)
    condition_a_lhs = scaled_mu * (-term1 + term2) / denominator

    if condition_a_lhs <= np.sqrt(scaled_L ** 2 + 1.0):
        numerator_a = lambda_value + np.sqrt(
            (term1 ** 2 + scaled_L ** 2 * (lambda_value - 2.0 * (scaled_mu + 1.0)) ** 2)
            / (scaled_L ** 2 + 1.0)
        )
        return (numerator_a / (2.0 * (scaled_mu + 1.0))) ** 2

    threshold_b = (
        2.0 * (scaled_mu + 1.0) * (scaled_L + 1.0)
        * (scaled_mu + scaled_mu * scaled_L ** 2 - scaled_L ** 2 - 2.0 * scaled_mu * scaled_L - 1.0)
    ) / (
        2.0 * scaled_mu ** 2 - scaled_mu + scaled_mu * scaled_L ** 3 - scaled_L ** 3
        - 3.0 * scaled_mu * scaled_L ** 2 - scaled_L ** 2
        - 2.0 * scaled_mu ** 2 * scaled_L - scaled_mu * scaled_L - scaled_L - 1.0
    )
    if (
        scaled_L < 1.0
        and scaled_mu > (scaled_L ** 2 + 1.0) / ((scaled_L - 1.0) ** 2)
        and lambda_value <= threshold_b
    ):
        return abs(1.0 - lambda_value * (scaled_L + scaled_mu) / ((scaled_mu + 1.0) * (scaled_L + 1.0))) ** 2

    term3 = lambda_value * (scaled_L ** 2 + 1.0) - 2.0 * scaled_mu * (lambda_value + scaled_L ** 2 - 1.0)
    term4 = lambda_value * (1.0 + 2.0 * scaled_mu + scaled_L ** 2) - 2.0 * (scaled_mu + 1.0) * (scaled_L ** 2 + 1.0)
    numerator_c = (2.0 - lambda_value) * term3 * term4
    denominator_c = 4.0 * scaled_mu * (scaled_L ** 2 + 1.0) * (
        2.0 * scaled_mu * (lambda_value + scaled_L ** 2 - 1.0)
        - (2.0 - lambda_value) * (1.0 - scaled_L ** 2)
    )
    return numerator_c / denominator_c

rho_theory = rho_theoretical(mu, L, lambda_value, gamma)

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
{cite}`ryu2020operatorsplittingperformance{Theorem 4.3}`, i.e.,

```{math}
\|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{math}
\rho =
\begin{cases}
\left(\dfrac{\lambda +
\sqrt{\dfrac{\left(2(\lambda-1)\tilde{\mu}+\lambda-2\right)^2
+\tilde{L}^2\left(\lambda-2(\tilde{\mu}+1)\right)^2}{\tilde{L}^2+1}}}
{2(\tilde{\mu}+1)}\right)^2,
& \text{if } (a),
\\
\left|1-\lambda\dfrac{\tilde{L}+\tilde{\mu}}{(\tilde{\mu}+1)(\tilde{L}+1)}\right|^2,
& \text{if } (b),
\\
\rho_{3},
& \text{otherwise},
\end{cases}
```

with

```{math}
\begin{aligned}
(a)\quad &
\dfrac{\tilde{\mu}\left(-2(\lambda-1)\tilde{\mu}-\lambda+2
+\tilde{L}^2\left(\lambda-2(\tilde{\mu}+1)\right)\right)}
{\sqrt{\left(2(\lambda-1)\tilde{\mu}+\lambda-2\right)^2
+\tilde{L}^2\left(\lambda-2(\tilde{\mu}+1)\right)^2}}
\le \sqrt{\tilde{L}^2+1},
\\
(b)\quad &
\tilde{L}<1
\text{ and } \tilde{\mu}>\dfrac{\tilde{L}^2+1}{(\tilde{L}-1)^2}
\text{ and } \lambda \le
\dfrac{2(\tilde{\mu}+1)(\tilde{L}+1)
\left(\tilde{\mu}+\tilde{\mu}\tilde{L}^2-\tilde{L}^2-2\tilde{\mu}\tilde{L}-1\right)}
{2\tilde{\mu}^2-\tilde{\mu}+\tilde{\mu}\tilde{L}^3-\tilde{L}^3-3\tilde{\mu}\tilde{L}^2-\tilde{L}^2-2\tilde{\mu}^2\tilde{L}-\tilde{\mu}\tilde{L}-\tilde{L}-1},
\end{aligned}
```

and

```{math}
\begin{aligned}
\rho_{3} &= \dfrac{(2-\lambda)
\left(\lambda(\tilde{L}^2+1)-2\tilde{\mu}(\lambda+\tilde{L}^2-1)\right)
\left(\lambda(1+2\tilde{\mu}+\tilde{L}^2)-2(\tilde{\mu}+1)(\tilde{L}^2+1)\right)}
{4\tilde{\mu}(\tilde{L}^2+1)
\left(2\tilde{\mu}(\lambda+\tilde{L}^2-1)-(2-\lambda)(1-\tilde{L}^2)\right)},\\
\tilde{\mu} &= \gamma\mu,\\
\tilde{L} &= \gamma L,
\end{aligned}
```

and

```{math}
x^\star \in \zer(G_1 + G_2).
```

Sweeping over 100 values of {math}`\lambda` on {math}`0 < \lambda < 2` (with
{math}`\mu=1`, {math}`L=1`, {math}`\gamma=1`) gives the plot below, with the
theoretical rate from {cite}`ryu2020operatorsplittingperformance{Theorem 4.3}`
in black and AutoLyap certificates as blue dots.

```{image} ../../_static/douglas_rachford_maximally_monotone_lipschitz_plus_strongly_monotone_rho_vs_lambda.svg
:alt: Douglas-Rachford rho versus lambda for maximally-monotone-Lipschitz plus strongly-monotone setting with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
