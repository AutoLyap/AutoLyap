# The Chambolle--Pock method: Convex

## Problem setup

Suppose {math}`f_1,f_2:\calH \to \reals\cup\set{\pm\infty}` are proper, convex, and lower semicontinuous.
In the identity-operator case, the Chambolle--Pock method
{cite}`chambolle2011firstorderprimal` solves

```{math}
\minimize_{y\in\calH}\; f_1(y)+f_2(y)
```

by solving the inclusion

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

where {math}`\tau,\sigma\in\reals_{++}` are primal and dual step sizes, respectively, and
{math}`\theta\in\reals` is a relaxation parameter.

For this example, the performance measure is the squared fixed-point residual
{math}`\|\bx^{k+1}-\bx^k\|^2` with {math}`\bx^k=(x^k,y^k)`.
If it is zero, then {math}`x^k` solves the inclusion above.

## Model the problem in AutoLyap and certify summability

- {math}`f_1,f_2` are modeled by
  {py:class}`Convex <autolyap.problemclass.Convex>`.
- The problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`ChambollePock <autolyap.algorithms.ChambollePock>`.
- Fixed-point-residual parameters are obtained with
  {py:meth}`IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual <autolyap.IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual>`.
- Feasibility is checked with
  {py:meth}`IterationIndependent.search_lyapunov <autolyap.IterationIndependent.search_lyapunov>`
  at {math}`\rho=1`, using the history parameter {math}`h` and overlap
  parameter {math}`\alpha`.

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import ChambollePock
from autolyap.problemclass import Convex, InclusionProblem

tau = 1.0
sigma = 1.0
theta = 1.0
h = 1
alpha = 1

problem = InclusionProblem([
    Convex(),  # f_1
    Convex(),  # f_2
])
algorithm = ChambollePock(tau=tau, sigma=sigma, theta=theta)

MOSEK_PARAMS = {
    "intpntCoTolPfeas": 1e-6,
    "intpntCoTolDfeas": 1e-6,
    "intpntCoTolRelGap": 1e-6,
    "intpntMaxIterations": 10000,
}
solver_options = SolverOptions(
    backend="mosek_fusion",
    mosek_params=MOSEK_PARAMS,
)
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

P, p, T, t = IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual(
    algorithm,
    h=h,
    alpha=alpha,
)

result = IterationIndependent.search_lyapunov(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    rho=1.0,
    h=h,
    alpha=alpha,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate for the selected parameters.")

print("Feasible certificate found.")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: fixed contraction parameter (`1.0` in this setup).
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

In this setup, feasibility certifies that

```{eval-rst}
.. math::
   \bigl(\|\bx^{k+1}-\bx^k\|^2\bigr)_{k\in\naturals}
```

is summable, and therefore,

```{eval-rst}
.. math::
   \|\bx^{k+1}-\bx^k\|^2 \to 0 \quad \textup{ as } \quad k\to\infty.
```

Sweeping over multiple values of {math}`\tau=\sigma\in(1,2)` and
{math}`\theta\in(0,3/2)` for several {math}`(h,\alpha)` settings gives the
layered region plot below.

```{image} ../../_static/chambolle_pock_fixed_point_residual_layers.svg
:alt: Feasible regions for Chambolle--Pock fixed-point residual summability in the (tau=sigma, theta) plane for several (h, alpha) settings.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
