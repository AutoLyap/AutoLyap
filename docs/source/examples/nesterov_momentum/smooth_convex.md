# The gradient method with constant Nesterov momentum: Smooth and convex

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x) \quad \iff \quad \text{find } x \in \calH \;\text{ such that }\; 0 \in \partial f(x) = \set{\nabla f(x)},
```

where {math}`f : \calH \to \reals` is convex and {math}`L`-smooth with
{math}`L>0`.

For initial points {math}`x^{-1}, x^0 \in \calH`, momentum
{math}`\delta \in \reals`, and step size {math}`\gamma \in \reals_{++}`, the gradient method with constant Nesterov-momentum is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
y^k &= x^k + \delta(x^k - x^{k-1}), \\
x^{k+1} &= y^k - \gamma \nabla f(y^k).
\end{aligned}
\right.
```

In this example, we search for a feasible Lyapunov certificate with
{math}`\rho=1` and condition (C4) enabled such that

```{eval-rst}
.. math::
   f(x^k) - f(x^\star) \in o\!\left(\frac{1}{k}\right) \quad \textup{ as } \quad k\to\infty,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

## Model the problem in AutoLyap and certify sublinear convergence

- {math}`f` is modeled by
  {py:class}`SmoothConvex <autolyap.problemclass.SmoothConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`GradientNesterovMomentum <autolyap.algorithms.GradientNesterovMomentum>`.
- Sublinear function-value parameters are obtained with
  {py:meth}`IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality <autolyap.IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality>`.
- Feasibility at fixed {math}`\rho=1` is checked with
  {py:meth}`IterationIndependent.search_lyapunov <autolyap.IterationIndependent.search_lyapunov>` and `remove_C4=False`.

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import IterationIndependent, SolverOptions
from autolyap.algorithms import GradientNesterovMomentum
from autolyap.problemclass import InclusionProblem, SmoothConvex

L = 1.0
gamma = 1.0
delta = 0.5

problem = InclusionProblem([SmoothConvex(L)])
algorithm = GradientNesterovMomentum(gamma=gamma, delta=delta)

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

# Build V(P, p, k) and R(T, t, k) for function-value suboptimality.
P, p, T, t = IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality(
    algorithm,
    tau=0,
)

result = IterationIndependent.search_lyapunov(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    rho=1.0,
    remove_C4=False,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate for this (gamma, delta) pair.")

print("Feasible certificate found with C4 enabled.")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: the fixed contraction parameter (`1.0` in this sublinear setup).
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

When the certificate is feasible, the certified function-value convergence is

```{math}
f(x^k) - f(x^\star) \in o\!\left(\frac{1}{k}\right)
\quad \textup{ as } \quad k\to\infty,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

Equivalently,

```{math}
\lim_{k \to \infty} k\bigl(f(x^k)-f(x^\star)\bigr)=0.
```

Sweeping over multiple values of {math}`\gamma \in (0,4)` and
{math}`\delta \in (-1,1)` with {math}`L=1` gives the plot below, where each
blue dot denotes a parameter pair with a feasible certificate.

```{image} ../../_static/nesterov_momentum_smooth_convex.svg
:alt: Certified smooth-convex region for constant Nesterov momentum in the (gamma, delta) plane.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
