# The gradient method: Smooth and strongly convex

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is {math}`\mu`-strongly convex and
{math}`L`-smooth, with {math}`0 < \mu < L`.

For an initial point {math}`x^0 \in \calH` and step size
{math}`0 < \gamma < 2/L`, the gradient update is

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = x^k - \gamma \nabla f(x^k).
```

In this example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \Argmin_{x \in \calH} f(x).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`GradientMethod <autolyap.algorithms.GradientMethod>`.
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
from autolyap.algorithms import GradientMethod
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex

mu = 1.0
L = 4.0
gamma = 0.2

problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
algorithm = GradientMethod(gamma=gamma)
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
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]
rho_theory = max(gamma * L - 1.0, 1.0 - gamma * mu) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

The computed value `rho (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical rate expression for gradient methods
{cite}`Polyak1963GradientUSSR`:

```{math}
\|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad k\to\infty, \qquad
\rho = \max\{|1-\gamma L|,\;|1-\gamma\mu|\}^2.
```

Equivalently,

```{math}
\|x^k - x^\star\| \in \mathcal{O}\!\left(\max\{|1-\gamma L|,\;|1-\gamma\mu|\}^k\right)
\quad \textup{ as } \quad k\to\infty.
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 2/L` gives
the plot below, with the theoretical rate in black and AutoLyap certificates
as blue dots.

```{image} ../../_static/gradient_method_rho_vs_gamma.svg
:alt: Gradient-method rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
