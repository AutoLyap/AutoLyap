# The heavy-ball method: Gradient-dominated and smooth

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x) \quad \iff \quad \text{find } x \in \calH \;\text{ such that }\; 0 \in \partial f(x) = \set{\nabla f(x)},
```

where {math}`f : \calH \to \reals` is {math}`\mu_{\textup{gd}}`-gradient
dominated and {math}`L`-smooth with {math}`0<\mu_{\textup{gd}}\le L`.

For initial points {math}`x^{-1}, x^0 \in \calH`, momentum
{math}`\delta \in \reals`, and step size {math}`\gamma \in \reals_{++}`,
the heavy-ball update {cite}`Polyak1964HeavyBall` is

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = x^k - \gamma \nabla f(x^k) + \delta(x^k - x^{k-1}).
```

In this example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   f(x^k) - f(x^\star) = O(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \Argmin_{x \in \calH} f(x).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f` is modeled as an intersection of
  {py:class}`GradientDominated <autolyap.problemclass.GradientDominated>` and
  {py:class}`Smooth <autolyap.problemclass.Smooth>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`HeavyBallMethod <autolyap.algorithms.HeavyBallMethod>`.
- Function-value parameters are obtained with
  {py:meth}`IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality <autolyap.IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality>`.
- The contraction factor is searched with
  {py:meth}`IterationIndependent.LinearConvergence.bisection_search_rho <autolyap.IterationIndependent.LinearConvergence.bisection_search_rho>`.

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import SolverOptions
from autolyap.algorithms import HeavyBallMethod
from autolyap.problemclass import GradientDominated, InclusionProblem, Smooth
from autolyap.iteration_independent import IterationIndependent

mu_gd = 0.5
L = 1.0
gamma = 1.0
delta = 0.5

problem = InclusionProblem(
    [
        [GradientDominated(mu_gd=mu_gd), Smooth(L=L)],  # f in intersection
    ]
)
algorithm = HeavyBallMethod(gamma=gamma, delta=delta)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

# Build V(P, p, k) and R(T, t, k) for function-value suboptimality analysis.
P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality(
    algorithm,
    tau=0,
)

result = IterationIndependent.LinearConvergence.bisection_search_rho(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

print(f"rho (AutoLyap): {result['rho']:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

Sweeping over 100 values of {math}`\gamma` on {math}`(0,2.30]` and 100 values
of {math}`\delta` on {math}`[-1,1]` with {math}`\mu_{\textup{gd}}=0.5` and
{math}`L=1` gives the plot below. Each dot is a feasible parameter pair, and
the color encodes the certified contraction factor {math}`\rho`.

```{image} ../../_static/heavy_ball_gradient_dominated_smooth.svg
:alt: Heavy-ball feasible (gamma, delta) pairs for gradient-dominated smooth objectives, with color showing rho.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
