# The Douglas--Rachford method

## Problem setup

Consider the monotone inclusion

```{math}
\text{find } x \in \calH \text{ such that } 0 \in G_1(x) + G_2(x),
```

where:

- {math}`G_1 : \calH \rightrightarrows \calH` is maximally monotone,
- {math}`G_2 : \calH \to \calH` is both {math}`\mu`-strongly monotone and
  {math}`L`-Lipschitz, with {math}`0 < \mu \le L`.

For an initial point {math}`x^0 \in \calH`, step size
{math}`\gamma \in \reals_{++}`, and relaxation
{math}`\lambda \in \reals`, the Douglas--Rachford update is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
v^k &= J_{\gamma G_1}(x^k), \\
w^k &= J_{\gamma G_2}(2v^k - x^k), \\
x^{k+1} &= x^k + \lambda (w^k - v^k).
\end{aligned}
\right.
```

## Run the iteration-independent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
import numpy as np

from autolyap import SolverOptions
from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    InclusionProblem,
    LipschitzOperator,
    MaximallyMonotone,
    StronglyMonotone,
)
from autolyap.iteration_independent import IterationIndependent


mu = 1.0
L = 2.0
gamma = 1.0
lambda_value = 2.0

problem = InclusionProblem(
    [MaximallyMonotone(), [StronglyMonotone(mu=mu), LipschitzOperator(L=L)]]
)
algorithm = DouglasRachford(gamma=gamma, lambda_value=lambda_value, type="operator")
solver_options = SolverOptions(backend="mosek_fusion")

P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
    algorithm
)

result = IterationIndependent.LinearConvergence.bisection_search_rho(
    problem,
    algorithm,
    P,
    T,
    S_equals_T=True,
    s_equals_t=True,
    remove_C3=True,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]

alpha = lambda_value / 2.0
delta = np.sqrt(1.0 - (4.0 * gamma * mu) / (1.0 + 2.0 * gamma * mu + (gamma * L) ** 2))
rho_theory = (abs(1.0 - alpha) + alpha * delta) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

The computed value `rho (AutoLyap)` matches the theoretical rate expression in
Theorem 6.5 of {cite}`Giselsson2017TightDouglasRachford`, i.e.,

```{math}
\|x^k - x^\star\|^2 = O(\rho^k), \qquad
\rho = \left(|1-\alpha| + \alpha \delta\right)^2,
```

with

```{math}
\alpha = \frac{\lambda}{2}, \qquad
\delta = \sqrt{1 - \frac{4\gamma\mu}{1 + 2\gamma\mu + (\gamma L)^2}},
```

where 


```{math}
x^\star \in \zer(G_1 + G_2).
```


Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 5` (with
{math}`\mu=1`, {math}`L=2`, {math}`\lambda=2`) gives the plot below, with the
theoretical rate in black and AutoLyap certificates as blue dots.

```{image} ../_static/douglas_rachford_operator_giselsson_thm65_rho_vs_gamma.svg
:alt: Douglas-Rachford rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
