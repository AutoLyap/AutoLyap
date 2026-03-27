# The Douglas--Rachford method: Maximally monotone + strongly monotone/Lipschitz

## Problem setup

Consider the monotone inclusion

```{math}
\text{find } x \in \calH \text{ such that } 0 \in G_1(x) + G_2(x),
```

where:

- {math}`G_1 : \calH \rightrightarrows \calH` is maximally monotone,
- {math}`G_2 : \calH \to \calH` is both {math}`\mu`-strongly monotone and
  {math}`L`-Lipschitz, with {math}`0 < \mu \le L`.

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
  {py:class}`MaximallyMonotone <autolyap.problemclass.MaximallyMonotone>`.
- {math}`G_2` is modeled as an intersection of
  {py:class}`StronglyMonotone <autolyap.problemclass.StronglyMonotone>` and
  {py:class}`LipschitzOperator <autolyap.problemclass.LipschitzOperator>`.
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
L = 2.0
gamma = 1.0
lambda_value = 2.0

problem = InclusionProblem(
    [
        MaximallyMonotone(),  # G1
        [StronglyMonotone(mu=mu), LipschitzOperator(L=L)],  # G2 in intersection
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

alpha = lambda_value / 2.0
delta = np.sqrt(1.0 - (4.0 * gamma * mu) / (1.0 + 2.0 * gamma * mu + (gamma * L) ** 2))
rho_theory = (abs(1.0 - alpha) + alpha * delta) ** 2

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
{cite}`Giselsson2017TightDouglasRachford{Theorem 6.5}`, i.e.,

```{math}
\|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{math}
\rho = \left(|1-\alpha| + \alpha \delta\right)^2, \quad
\alpha = \frac{\lambda}{2}, \quad
\delta = \sqrt{1 - \frac{4\gamma\mu}{1 + 2\gamma\mu + (\gamma L)^2}},
```

and

```{math}
x^\star \in \zer(G_1 + G_2).
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 5` (with
{math}`\mu=1`, {math}`L=2`, {math}`\lambda=2`) gives the plot below, with the
theoretical rate in black and AutoLyap certificates as blue dots.

```{image} ../../_static/douglas_rachford_maximally_monotone_plus_strongly_monotone_lipschitz_rho_vs_gamma.svg
:alt: Douglas-Rachford rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
