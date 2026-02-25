# The Douglas--Rachford method: Smooth and strongly convex + convex

## Problem setup

Consider the composite minimization problem

```{math}
\minimize_{x \in \calH}\; f_1(x)+f_2(x),
```

where:

- {math}`f_1 : \calH \to \reals` is {math}`\mu`-strongly convex and
  {math}`L`-smooth, with {math}`0 < \mu < L`,
- {math}`f_2 : \calH \to \reals \cup \set{\pm\infty}` is proper, convex, and
  lower semicontinuous.

Equivalently, we solve

```{math}
\text{find } x \in \calH \text{ such that } 0 \in \partial f_1(x) + \partial f_2(x).
```

For an initial point {math}`x^0 \in \calH`, step size
{math}`\gamma \in \reals_{++}`, and relaxation
{math}`\lambda \in \reals`, the Douglas--Rachford update is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
v^k &= \prox_{\gamma f_1}(x^k), \\
w^k &= \prox_{\gamma f_2}(2v^k - x^k), \\
x^{k+1} &= x^k + \lambda (w^k - v^k).
\end{aligned}
\right.
```

In this example, we fix {math}`\mu=1`, {math}`L=2`, {math}`\lambda=1`, sweep
{math}`\gamma \in (0,5]`, and search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad  k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \zer\bigl(\partial f_1 + \partial f_2\bigr).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f_1` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- {math}`f_2` is modeled by
  {py:class}`Convex <autolyap.problemclass.Convex>`.
- The full inclusion is built with
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`DouglasRachford <autolyap.algorithms.DouglasRachford>` with
  `type="function"`.
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

from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex

mu = 1.0
L = 2.0
gamma = 1.0
lambda_value = 1.0

problem = InclusionProblem(
    [
        SmoothStronglyConvex(mu=mu, L=L),  # f1
        Convex(),  # f2
    ]
)
algorithm = DouglasRachford(
    gamma=gamma,
    lambda_value=lambda_value,
    type="function",
)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
    algorithm
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
    tol=1e-3,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]

alpha = lambda_value / 2.0
delta = max(
    (gamma * L - 1.0) / (gamma * L + 1.0),
    (1.0 - gamma * mu) / (1.0 + gamma * mu),
)
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
the rate expression in
{cite}`giselsson2017linearconvergencemetric{Theorem 2}` for this setting, i.e.,

```{math}
\rho = \left(|1-\alpha| + \alpha \delta\right)^2, \quad
\alpha = \frac{\lambda}{2}, \quad
\delta = \max\left\{
\frac{\gamma L - 1}{\gamma L + 1},
\frac{1-\gamma\mu}{1+\gamma\mu}
\right\}.
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 5` (with
{math}`\mu=1`, {math}`L=2`, {math}`\lambda=1`) gives the plot below, with the
theoretical rate in black and AutoLyap certificates as blue dots.

```{image} ../../_static/douglas_rachford_smooth_strongly_convex_plus_convex_rho_vs_gamma.svg
:alt: Douglas-Rachford rho versus gamma for smooth strongly-convex plus convex splitting, with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
