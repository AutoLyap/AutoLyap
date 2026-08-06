<!-- SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# The triple-momentum method

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is {math}`\mu`-strongly convex and
{math}`L`-smooth, with {math}`0 < \mu < L`.

Let {math}`q=\mu/L` and define

```{math}
\alpha_{\mathrm{tm}} = \frac{2-\sqrt{q}}{L}, \qquad
\beta_{\mathrm{tm}} = \frac{(1-\sqrt{q})^2}{1+\sqrt{q}}, \qquad
\gamma_{\mathrm{tm}} =
\frac{(1-\sqrt{q})^2}{(2-\sqrt{q})(1+\sqrt{q})}.
```

For initial points {math}`x^{-1},x^0\in\calH`, the triple-momentum method
{cite}`vanscoy2018fastestknownglobally` is given by

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
y^k &= x^k + \gamma_{\mathrm{tm}}(x^k-x^{k-1}), \\
x^{k+1} &= x^k + \beta_{\mathrm{tm}}(x^k-x^{k-1})
             - \alpha_{\mathrm{tm}}\nabla f(y^k).
\end{aligned}
\right.
```
In this
example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|y^k-x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where {math}`x^\star\in\Argmin_{x\in\calH} f(x)`.

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`TripleMomentum <autolyap.algorithms.TripleMomentum>`.
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
from autolyap.algorithms import TripleMomentum
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex

mu = 1.0
L = 4.0

problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L)])
algorithm = TripleMomentum(mu=mu, L=L)
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
rho_theory = (1.0 - (mu / L) ** 0.5) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

The computed value `rho (AutoLyap)` matches, up to solver numerical tolerances,
the theoretical rate expression for the triple-momentum method
{cite}`vanscoy2018fastestknownglobally{Theorem 1 and Corollary 1}`:

```{math}
\|y^k-x^\star\|^2 \in \mathcal{O}(\rho^k) \quad \textup{ as } \quad k\to\infty,
\qquad
\rho = \left(1-\sqrt{\frac{\mu}{L}}\right)^2.
```

Equivalently,

```{math}
\|y^k-x^\star\|
\in \mathcal{O}\!\left(\left(1-\sqrt{\frac{\mu}{L}}\right)^k\right)
\quad \textup{ as } \quad k\to\infty.
```

Sweeping over 100 values of {math}`L` on {math}`(1,100]`, with {math}`\mu=1`
fixed, gives the plot below. The theoretical rate is shown in black and the
MOSEK-backed AutoLyap certificates as blue dots.

```{image} ../_static/triple_momentum_rho_vs_smoothness.svg
:alt: Triple-momentum rho versus L with the theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
