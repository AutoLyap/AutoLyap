# The Davis--Yin three-operator splitting method

## Problem setup

Consider the composite minimization problem

```{math}
\minimize_{x \in \calH} f_1(x) + f_2(x) + f_3(x),
```

where:

- {math}`f_1 : \calH \to \reals` is convex and {math}`L_1`-smooth, with {math}`L_1>0`,
- {math}`f_2 : \calH \to \reals` is {math}`\mu_2`-strongly convex and {math}`L_2`-smooth,
  with {math}`0<\mu_2\le L_2`,
- {math}`f_3 : \calH \to \reals \cup \set{\pm\infty}` is proper, convex, and lower semicontinuous.

Equivalently, we solve

```{math}
\text{find } x \in \calH \text{ such that } 0 \in \partial f_1(x) + \nabla f_2(x) + \partial f_3(x).
```

For an initial point {math}`x^0 \in \calH`, step size
{math}`\gamma \in \reals_{++}`, and relaxation
{math}`\lambda \in \reals`, the Davis--Yin method
{cite}`davis2017threeoperatorsplitting` is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
v^k &= \prox_{\gamma f_1}(x^k), \\
w^k &= \prox_{\gamma f_3}\!\left(2v^k - x^k - \gamma \nabla f_2(v^k)\right), \\
x^{k+1} &= x^k + \lambda (w^k - v^k).
\end{aligned}
\right.
```

In this example, we fix
{math}`\mu_2 = 1`, {math}`L_2 = 2`, {math}`\lambda = 1`,
{math}`\gamma = 1/L_2`, sweep {math}`L_1 \in (0, 40]`, and search for the
smallest contraction factor {math}`\rho\in[0,1)` provable using AutoLyap such that

```{eval-rst}
.. math::
   \|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where

```{eval-rst}
.. math::
   x^\star \in \Argmin_{x \in \calH} f_1(x)+f_2(x)+f_3(x).
```

## Model the problem in AutoLyap and search for the smallest rho

- {math}`f_1` is modeled by
  {py:class}`SmoothConvex <autolyap.problemclass.SmoothConvex>`.
- {math}`f_2` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- {math}`f_3` is modeled by
  {py:class}`Convex <autolyap.problemclass.Convex>`.
- The inclusion is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`DavisYin <autolyap.algorithms.DavisYin>`.
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
from autolyap.algorithms import DavisYin
from autolyap.problemclass import Convex, InclusionProblem, SmoothConvex, SmoothStronglyConvex

L1 = 1.0
mu2 = 1.0
L2 = 2.0
lambda_value = 1.0
gamma = 1.0 / L2

problem = InclusionProblem(
    [
        SmoothConvex(L=L1),  # f1
        SmoothStronglyConvex(mu=mu2, L=L2),  # f2
        Convex(),  # f3
    ]
)
algorithm = DavisYin(gamma=gamma, lambda_value=lambda_value)
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
rho_davis_yin = 1.0 - mu2 / (L2 * (1.0 + L1 / L2) ** 2)
rho_pedregosa_gidel = 1.0 - min(mu2 / L2, 1.0 / (1.0 + L1 / L2))

print(f"rho (AutoLyap):         {rho_autolyap:.8f}")
print(f"rho (Davis-Yin bound):  {rho_davis_yin:.8f}")
print(f"rho (Pedregosa-Gidel):  {rho_pedregosa_gidel:.8f}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["rho"]`: certified contraction factor when feasible.
- `result["certificate"]`: Lyapunov certificate matrices/scalars.

For {math}`\lambda=1` and {math}`\gamma=1/L_2`, we compare against:

```{math}
\rho_{\text{DY}} = 1 - \frac{\mu_2}{L_2\left(1+L_1/L_2\right)^2},
```

This is the specialization of
{cite}`davis2015threeoperatorsplitting{Theorem D.6, case 6}` under:

- case-6 assumptions (`B` Lipschitz and `C` strongly monotone),
- parameter choice {math}`\lambda=1`,
- limiting choices {math}`\varepsilon \to 1` and {math}`\eta \to 1/2`,
- identifications {math}`L_B=L_1`, {math}`\mu_C=\mu_2`, and
  {math}`\gamma=1/L_2`.

and

```{math}
\rho_{\text{PG}} = 1 - \min\!\left\{\frac{\mu_2}{L_2},\;\frac{1}{1+L_1/L_2}\right\},
```

This matches the non-adaptive TOS specialization of
{cite}`pedregosa2018adaptivethreeoperator{Eq. (16), Theorem 3}`:

- define {math}`\rho=\mu_2\min\{\gamma,\tau/L_2\}` and
  {math}`\sigma=1/(1+\gamma L_1)`,
- use {math}`\tau=1` and {math}`\gamma=1/L_2`,
- then {math}`\rho=\mu_2/L_2` and {math}`\sigma=1/(1+L_1/L_2)`,
  which yields {math}`1-\min\{\rho,\sigma\}`.

Sweeping over 100 values of {math}`L_1` on {math}`(0, 40]` gives the plot
below, with the two theoretical expressions as lines and AutoLyap certificates
as blue dots.

```{image} ../_static/davis_yin_three_operator_rho_vs_l1.svg
:alt: Davis-Yin three-operator rho versus L1 with two theoretical lines and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
