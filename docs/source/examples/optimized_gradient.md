# The optimized gradient method

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is convex and {math}`L`-smooth with
{math}`L>0`.

For initial points {math}`x^0, y^0 \in \calH` and iteration budget
{math}`K \in \mathbb{N}`, the optimized gradient method {cite}`kim2015optimizedfirstorder` is given by

```{math}
(\forall k \in \llbracket 0, K-1 \rrbracket)\quad
\left[
\begin{aligned}
    y^{k+1} &= x^k - \frac{1}{L}\nabla f(x^k), \\
    x^{k+1} &= y^{k+1}
    + \frac{\theta_k - 1}{\theta_{k+1}}(y^{k+1} - y^k)
    + \frac{\theta_k}{\theta_{k+1}}(y^{k+1} - x^k).
\end{aligned}
\right.
```

```{math}
\theta_k =
\begin{cases}
    1, & \text{if } k = 0, \\
    \dfrac{1 + \sqrt{1 + 4\theta_{k-1}^2}}{2},
    & \text{if } k \in \llbracket 1, K-1 \rrbracket, \\
    \dfrac{1 + \sqrt{1 + 8\theta_{k-1}^2}}{2},
    & \text{if } k = K.
\end{cases}
```

In this example, we search for the smallest AutoLyap certificate constant
{math}`c_K` such that

```{math}
f(x^K)-f(x^\star)\le c_K\|x^0-x^\star\|^2,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

## Model the problem in AutoLyap and search for the smallest c_K

- {math}`f` is modeled by
  {py:class}`SmoothConvex <autolyap.problemclass.SmoothConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`OptimizedGradientMethod <autolyap.algorithms.OptimizedGradientMethod>`.
- Initial-horizon parameters are obtained with
  {py:meth}`IterationDependent.get_parameters_distance_to_solution <autolyap.IterationDependent.get_parameters_distance_to_solution>`.
- Final-horizon function-value parameters are obtained with
  {py:meth}`IterationDependent.get_parameters_function_value_suboptimality <autolyap.IterationDependent.get_parameters_function_value_suboptimality>`.
- The certificate constant is computed with
  {py:meth}`IterationDependent.search_lyapunov <autolyap.IterationDependent.search_lyapunov>`.

## Run the iteration-dependent analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import IterationDependent, SolverOptions
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex

L = 1.0
K = 5

problem = InclusionProblem([SmoothConvex(L)])
algorithm = OptimizedGradientMethod(L=L, K=K)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

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

result = IterationDependent.search_lyapunov(
    problem,
    algorithm,
    K,
    Q_0,
    Q_K,
    q_0=q_0,
    q_K=q_K,
    solver_options=solver_options,
)

if result["status"] != "feasible":
    raise RuntimeError("No feasible chained Lyapunov certificate for this setup.")

c_K = result["c_K"]
theta_K = algorithm.compute_theta(K, K)
c_K_theory = L / (2.0 * theta_K ** 2)

print(f"c_K (AutoLyap): {c_K:.6e}")
print(f"c_K (theory):   {c_K_theory:.6e}")
```

What to inspect in `result`:

- `result["status"]`: `feasible`, `infeasible`, or `not_solved`.
- `result["solve_status"]`: raw backend status.
- `result["c_K"]`: certified finite-horizon constant when feasible.
- `result["certificate"]`: chained Lyapunov certificate matrices/vectors.

The computed value `c_K (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical horizon-`K` expression:

```{math}
f(x^K) - f(x^\star) \le c_K\,\|x^0 - x^\star\|^2, \qquad
c_K = \frac{L}{2\theta_K^2}.
```

In particular,

```{math}
f(x^K) - f(x^\star) = O\!\left(\frac{1}{\theta_K^2}\right) = O\!\left(\frac{1}{K^2}\right).
```

Sweeping over {math}`K \in \llbracket 1, 100\rrbracket` gives the log-log plot
below, with the theoretical bound in black and AutoLyap certificates as blue
dots.

```{image} ../_static/optimized_gradient_method_c_vs_K_loglog.svg
:alt: Optimized-gradient c_K versus K in log-log scale with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
