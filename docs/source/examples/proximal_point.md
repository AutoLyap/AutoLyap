# The proximal point method

## Problem setup

Consider minimizing a {math}`\mu`-strongly convex function
{math}`f : \calH \to \reals`, with {math}`\mu > 0`; for an initial point
{math}`x^0 \in \calH` and step size {math}`\gamma \in \reals_{++}`, the proximal
point method updates as

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = \prox_{\gamma f}(x^k).
```

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import SolverOptions
from autolyap.algorithms import ProximalPoint
from autolyap.problemclass import InclusionProblem, StronglyConvex
from autolyap.iteration_independent import IterationIndependent

mu = 1.0
gamma = 0.6

problem = InclusionProblem([StronglyConvex(mu)])
algorithm = ProximalPoint(gamma=gamma)
solver_options = SolverOptions(backend="mosek_fusion")

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
rho_theory = (1.0 / (1.0 + gamma * mu)) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

The computed value `rho (AutoLyap)` matches the theoretical rate expression
given in {cite}`Rockafellar1976PPA`, i.e.,

```{math}
\|x^k - x^\star\|^2 = O(\rho^k), \qquad
\rho = \left(\frac{1}{1+\gamma\mu}\right)^2,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

Equivalently,

```{math}
\|x^k - x^\star\| = O\!\left(\left(\frac{1}{1+\gamma\mu}\right)^k\right).
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 5` gives
the plot below, with the theoretical rate in black and AutoLyap certificates
as blue dots.

```{image} ../_static/proximal_point_rho_vs_gamma.svg
:alt: Proximal-point rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
