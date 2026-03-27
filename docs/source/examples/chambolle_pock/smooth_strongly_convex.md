# The Chambolle--Pock method: Smooth and strongly convex

## Problem setup

Suppose {math}`f_1,f_2:\calH \to \reals` are {math}`\mu`-strongly convex and
{math}`L`-smooth with {math}`0<\mu<L`.
In the identity-operator case, the Chambolle--Pock method
{cite}`chambolle2011firstorderprimal` solves

```{math}
\minimize_{y\in\calH}\; f_1(y)+f_2(y)
```

via the inclusion

```{math}
\text{find } y\in\calH\;\text{such that}\; 0\in\partial f_1(y)+\partial f_2(y),
```

with iterations

```{math}
(\forall k\in\naturals)\quad
\left[
\begin{aligned}
x^{k+1} &= \prox_{\tau f_1}(x^k-\tau y^k), \\
y^{k+1} &= \prox_{\sigma f_2^*}\!\left(y^k + \sigma\left(x^{k+1}+\theta(x^{k+1}-x^k)\right)\right),
\end{aligned}
\right.
```

where {math}`\tau,\sigma\in\reals_{++}` are primal and dual step sizes,
respectively, and {math}`\theta\in\reals` is a relaxation parameter.

In this example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \zer\bigl(\partial f_1 + \partial f_2\bigr).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f_1,f_2` are modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`ChambollePock <autolyap.algorithms.ChambollePock>`.
- Distance-to-solution parameters are obtained with
  {py:meth}`IterationIndependent.LinearConvergence.get_parameters_distance_to_solution <autolyap.IterationIndependent.LinearConvergence.get_parameters_distance_to_solution>`
  using `i=1`.
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
from autolyap.algorithms import ChambollePock
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex

mu = 0.05
L = 50.0
tau = 1.6
sigma = tau
theta = 0.22

problem = InclusionProblem(
    [
        SmoothStronglyConvex(mu=mu, L=L),  # f_1
        SmoothStronglyConvex(mu=mu, L=L),  # f_2
    ]
)
algorithm = ChambollePock(tau=tau, sigma=sigma, theta=theta)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
    algorithm,
    i=1,
)

result = IterationIndependent.LinearConvergence.bisection_search_rho(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    tol=1e-3,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

print(f"rho (AutoLyap): {result['rho']:.6f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

Sweeping over 100 values of {math}`\tau=\sigma` on {math}`[0.5,1.75]` and 100
values of {math}`\theta` on {math}`[0,8]` with
{math}`f_1,f_2\in\mathcal{F}_{0.05,50}` gives the plot below.
Each dot is a feasible parameter pair, and the color encodes the certified
contraction factor {math}`\rho`.

```{image} ../../_static/chambolle_pock_smooth_strongly_convex_rho.svg
:alt: Certified Chambolle--Pock linear rates over (tau equals sigma, theta) for smooth strongly convex objectives.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
