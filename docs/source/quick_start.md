# Quick start

This page gives two end-to-end workflows:

1. Iteration-independent analysis with a bisection search on `rho`.
2. Iteration-dependent analysis with chained Lyapunov inequalities.

## Prerequisites

Install AutoLyap.

```bash
pip install autolyap
```

## Workflow

Typical usage has four steps:

1. Build an `InclusionProblem` from function and operator classes.
2. Pick an algorithm or your own subclass of `Algorithm`.
3. Select Lyapunov targets with helper constructors (`get_parameters_*`).
4. Solve the SDP and inspect `result["success"]`, the scalar (`rho` or `c`), and `result["certificate"]`.

## Iteration-independent example

```python
from autolyap.algorithms import GradientMethod
from autolyap import SolverOptions
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex
from autolyap.iteration_independent import IterationIndependent

mu = 1.0
L = 4.0
gamma = 0.2

problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
algorithm = GradientMethod(gamma=gamma)
solver_options = SolverOptions(backend="mosek_fusion")

# License-free options
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="SCS")

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

if not result["success"]:
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho = result["rho"]
certificate = result["certificate"]

rho_theory = max(gamma * L - 1.0, 1.0 - gamma * mu) ** 2
print(f"rho (AutoLyap): {rho:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

## Iteration-dependent example

For background on `OptimizedGradientMethod`, see {cite}`quick-kin2028ogm`.

```python
from autolyap.algorithms import OptimizedGradientMethod
from autolyap import SolverOptions
from autolyap.problemclass import InclusionProblem, SmoothConvex
from autolyap.iteration_dependent import IterationDependent

L = 1.0
K = 5

problem = InclusionProblem([SmoothConvex(L)])
algorithm = OptimizedGradientMethod(L=L, K=K)
solver_options = SolverOptions(backend="mosek_fusion")

# License-free options
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="SCS")

Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
    algorithm,
    0,
    i=1,
    j=1,
)
Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
    algorithm,
    K,
)

result = IterationDependent.verify_iteration_dependent_Lyapunov(
    problem,
    algorithm,
    K,
    Q_0,
    Q_K,
    q_0=q_0,
    q_K=q_K,
    solver_options=solver_options,
)

if not result["success"]:
    raise RuntimeError("No feasible chained Lyapunov certificate for this setup.")

c = result["c"]
certificate = result["certificate"]

Q_sequence = certificate["Q_sequence"]  # [Q_0, Q_1, ..., Q_K]
q_sequence = certificate["q_sequence"]  # [q_0, q_1, ..., q_K] or None

theta_K = algorithm._compute_theta(K, K)
c_theory = L / (2.0 * theta_K ** 2)

print(f"c (AutoLyap): {c:.6e}")
print(f"c (theory):   {c_theory:.6e}")
```

## What to inspect

- `success`: whether the selected backend found a feasible certificate.
- `rho` (iteration-independent) or `c` (iteration-dependent): the main scalar output.
- `certificate`: Matrices and vectors that parameterize the Lyapunov certificate.

## Next

- For more worked walkthroughs, see {doc}`examples`.
- For the full API, see {doc}`api_reference`.
- For full mathematical context, see the companion paper {cite}`quick-upadhyaya2025autolyap`.

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: quick-
```
