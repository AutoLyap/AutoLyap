# The proximal point method

## Problem setup

We consider one functional component:

- {math}`f` is {math}`\mu`-strongly convex, with {math}`\mu > 0`.
- Problem class: {py:class}`autolyap.problemclass.InclusionProblem` with
  {py:class}`autolyap.problemclass.StronglyConvex`.

## Run the iteration-independent analysis

```python
from autolyap.algorithms import ProximalPoint
from autolyap.problemclass import InclusionProblem, StronglyConvex
from autolyap.iteration_independent import IterationIndependent

mu = 1.0
gamma = 0.6

problem = InclusionProblem([StronglyConvex(mu)])
algorithm = ProximalPoint(gamma=gamma)

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
)

if not result["success"]:
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]
rho_theory = (1.0 / (1.0 + gamma * mu)) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

The theoretical rate expression for proximal point is classical; see {cite}`Rockafellar1976PPA`.

Hence, for the distance-to-solution target, the method satisfies

```{math}
\|x^k - x^\star\|^2 = O(\rho^k), \qquad
\rho = \left(\frac{1}{1+\gamma\mu}\right)^2.
```

Equivalently,

```{math}
\|x^k - x^\star\| = O\!\left(\left(\frac{1}{1+\gamma\mu}\right)^k\right).
```

## Optional: verify multiple step sizes

```python
import numpy as np

mu = 1.0
problem = InclusionProblem([StronglyConvex(mu)])
algorithm = ProximalPoint(gamma=1.0)

for gamma in np.linspace(0.05, 0.95, 10):
    algorithm.set_gamma(gamma)
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
    )
    if not result["success"]:
        raise RuntimeError(f"Failed to certify rho for gamma={gamma:.3f}")

    rho_theory = (1.0 / (1.0 + gamma * mu)) ** 2
    assert abs(result["rho"] - rho_theory) < 1e-5
```

## References

```{bibliography}
:filter: docname in docnames
```
