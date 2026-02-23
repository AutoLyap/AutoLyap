# The Malitsky--Tam forward-reflected-backward method

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

For an initial pair {math}`(x^{-1},x^0)\in\calH^2` and step size
{math}`\gamma \in \reals_{++}`, the Malitsky--Tam forward-reflected-backward method
{cite}`malitsky2020forwardbackwardsplitting` is given by

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = J_{\gamma G_2}\!\left(x^k - 2\gamma G_1(x^k) + \gamma G_1(x^{k-1})\right).
```

In this example, we fix {math}`\mu=1`, {math}`L=1`, sweep
{math}`\gamma \in (0,1]`, and search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad k\to\infty,
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
  {py:class}`MalitskyTamFRB <autolyap.algorithms.MalitskyTamFRB>`.
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
from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import MalitskyTamFRB
from autolyap.problemclass import (
    InclusionProblem,
    LipschitzOperator,
    MaximallyMonotone,
    StronglyMonotone,
)

mu = 1.0
L = 1.0
gamma = 0.2

problem = InclusionProblem(
    [
        [MaximallyMonotone(), LipschitzOperator(L=L)],  # G1
        StronglyMonotone(mu=mu),  # G2
    ]
)
algorithm = MalitskyTamFRB(gamma=gamma)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

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

if not (0.0 < gamma < 1.0 / (2.0 * L)):
    raise ValueError("Theorem 2.9 rate expression requires 0 < gamma < 1/(2L).")

epsilon = min(0.5 - gamma * L, 5.0 * mu * gamma)
alpha = min(1.0 + 4.0 * mu * gamma - 0.75 * epsilon, 1.0 + 0.5 * epsilon)
rho_theory = 1.0 / alpha

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

The computed value `rho (AutoLyap)` is compared against the rate
expression from
{cite}`malitsky2020forwardbackwardsplitting{Theorem 2.9}`.
Under the step-size condition {math}`0<\gamma<1/(2L)`, define

```{math}
\varepsilon = \min\!\left\{\frac{1}{2}-\gamma L,\;5\mu\gamma\right\},
\qquad
\alpha = \min\!\left\{1+4\mu\gamma-\frac{3}{4}\varepsilon,\;1+\frac{1}{2}\varepsilon\right\},
```

and set

```{math}
\rho_{\mathrm{MT}} = \frac{1}{\alpha}.
```

Then the theoretical bound is

```{math}
\|x^k - x^\star\|^2 = O(\rho_{\mathrm{MT}}^k) \quad \textup{ as } \quad k\to\infty,
```

where {math}`x^\star \in \zer(G_1+G_2)`.

Sweeping over 100 values of {math}`\gamma` on {math}`(0,1]` gives the plot
below, with the Theorem 2.9-derived expression (on
{math}`0<\gamma<1/(2L)`) in black and AutoLyap certificates as blue dots.

```{image} ../_static/malitsky_tam_frb_rho_vs_gamma.svg
:alt: Malitsky-Tam FRB rho versus gamma with theorem-derived line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
