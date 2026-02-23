# The gradient method: Gradient-dominated and smooth

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x) \quad \iff \quad \text{find } x \in \calH \;\text{ such that }\; 0 \in \partial f(x) = \set{\nabla f(x)},
```

where {math}`f : \calH \to \reals` is {math}`\mu_{\textup{gd}}`-gradient
dominated and {math}`L`-smooth with {math}`0<\mu_{\textup{gd}}\le L`. This
setting is not necessarily convex.

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
  {py:class}`GradientMethod <autolyap.algorithms.GradientMethod>`.
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
import math

from autolyap import SolverOptions
from autolyap.algorithms import GradientMethod
from autolyap.problemclass import GradientDominated, InclusionProblem, Smooth
from autolyap.iteration_independent import IterationIndependent


def rho_theory_theorem3_specialized(mu_gd: float, L: float, gamma: float) -> float:
    # Theorem 3 in abbaszadehpeivasti2023conditionslinearconvergence,
    # specialized to the L-smooth (possibly nonconvex) setting via mu = -L.
    threshold_1 = 1.0 / L
    threshold_2 = math.sqrt(3.0) / L

    if gamma < threshold_1:
        discriminant = 4.0 * L * L - (
            2.0 * L * (L + mu_gd) * mu_gd * gamma * (2.0 - L * gamma)
        )
        discriminant = max(discriminant, 0.0)
        numerator = mu_gd * (1.0 - L * gamma) + math.sqrt(discriminant)
        denominator = 2.0 * L + mu_gd
        return (numerator / denominator) ** 2

    if gamma <= threshold_2:
        return 1.0 - (mu_gd * gamma * (4.0 - (L * gamma) ** 2)) / (2.0 + mu_gd * gamma)

    numerator = (L * gamma - 1.0) ** 2
    denominator = numerator + mu_gd * gamma * (2.0 - L * gamma)
    return numerator / denominator


mu_gd = 0.5
L = 1.0
gamma = 1.0

problem = InclusionProblem(
    [
        [GradientDominated(mu_gd=mu_gd), Smooth(L=L)],  # f in intersection
    ]
)
algorithm = GradientMethod(gamma=gamma)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

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

rho_autolyap = result["rho"]
rho_theory = rho_theory_theorem3_specialized(mu_gd=mu_gd, L=L, gamma=gamma)

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

In the {math}`L`-smooth (possibly nonconvex) setting, we can specialize
{cite}`abbaszadehpeivasti2023conditionslinearconvergence{Theorem 3}` by setting
{math}`\mu=-L`. This gives the reference rate
{math}`\rho_{\textup{AdKZ}}`:

```{math}
\rho_{\textup{AdKZ}}=
\begin{cases}
\left(
\dfrac{
\mu_{\textup{gd}}(1-L\gamma)
+\sqrt{
4L^2-2L(L+\mu_{\textup{gd}})\mu_{\textup{gd}}\gamma(2-L\gamma)
}
}{2L+\mu_{\textup{gd}}}
\right)^2, & 0 < \gamma < \dfrac{1}{L},\\[1.0em]
1-\dfrac{\mu_{\textup{gd}}\gamma\bigl(4-(L\gamma)^2\bigr)}{2+\mu_{\textup{gd}}\gamma},
& \dfrac{1}{L} \le \gamma \le \dfrac{\sqrt{3}}{L},\\[1.0em]
\dfrac{(L\gamma-1)^2}{(L\gamma-1)^2+\mu_{\textup{gd}}\gamma(2-L\gamma)},
& \dfrac{\sqrt{3}}{L}<\gamma<\dfrac{2}{L}.
\end{cases}
```

We then sweep {math}`\gamma \in (0,2)` with
{math}`\mu_{\textup{gd}}=0.5` and {math}`L=1`.
In the plot below, the black curve is
{math}`\rho_{\textup{AdKZ}}`, while the blue markers are AutoLyap
certificates.

Numerically, AutoLyap gives tighter rates for
{math}`\gamma \in (1.74,2)`.
One plausible reason is that
{cite}`abbaszadehpeivasti2023conditionslinearconvergence{Theorem 3}` does not
use interpolation conditions at solutions
{math}`x^\star \in \Argmin_{x\in \calH} f(x)` when passing from (7) to (10),
which introduces a source of a priori conservatism.

```{image} ../../_static/gradient_method_gradient_dominated_smooth_rho_vs_gamma.svg
:alt: Gradient-method rho versus gamma for gradient-dominated smooth objectives, comparing AutoLyap and Theorem 3.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
