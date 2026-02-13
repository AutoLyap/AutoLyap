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

1. Build an {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>` from function and operator classes.
2. Pick an algorithm or your own subclass of {py:class}`Algorithm <autolyap.algorithms.Algorithm>`.
3. Select Lyapunov targets with helper constructors.
4. Solve the SDP and inspect `result["success"]`, the scalar (`rho` or `c_K`), and `result["certificate"]`.

## Iteration-independent example: The gradient method

Consider minimizing a function {math}`f : \calH \to \reals` that is {math}`\mu`-strongly convex and {math}`L`-smooth; for an initial point {math}`x^0 \in \calH` and step size {math}`0 < \gamma < 2/L`, the gradient method updates as

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = x^k - \gamma \nabla f(x^k).
```

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

# License-free option
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

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

The computed value `rho (AutoLyap)` matches the theoretical rate expression for
gradient methods; see {cite}`quick-Polyak1963GradientUSSR`:

```{math}
\|x^k - x^\star\|^2 = O(\rho^k), \qquad
\rho = \max\{|1-\gamma L|,\;|1-\gamma\mu|\}^2,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

Equivalently,

```{math}
\|x^k - x^\star\| = O\!\left(\max\{|1-\gamma L|,\;|1-\gamma\mu|\}^k\right).
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 2/L` gives
the plot below, with the theoretical rate in black and AutoLyap certificates
as blue dots.

```{image} _static/gradient_method_rho_vs_gamma.svg
:alt: Gradient-method rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## Iteration-dependent example: The optimized gradient method

For background on the optimized gradient method, see {cite}`quick-kin2028ogm`.

Consider minimizing a convex and {math}`L`-smooth function {math}`f : \calH \to \reals`, with {math}`L > 0`; for initial points {math}`x^0, y^0 \in \calH`, smoothness constant {math}`L \in \reals_{++}`, and iteration budget {math}`K \in \mathbb{N}`, the optimized gradient method updates as

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

# License-free option
#solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

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

if not result["success"]:
    raise RuntimeError("No feasible chained Lyapunov certificate for this setup.")

c_K = result["c_K"]
certificate = result["certificate"]

Q_sequence = certificate["Q_sequence"]  # [Q_0, Q_1, ..., Q_K]
q_sequence = certificate["q_sequence"]  # [q_0, q_1, ..., q_K] or None

theta_K = algorithm.compute_theta(K, K)
c_K_theory = L / (2.0 * theta_K ** 2)

print(f"c_K (AutoLyap): {c_K:.6e}")
print(f"c_K (theory):   {c_K_theory:.6e}")
```

The computed value `c_K (AutoLyap)` matches the theoretical horizon-`K`
expression:

```{math}
f(x^K) - f(x^\star) \le c_K\,\|x^0 - x^\star\|^2, \qquad
c_K = \frac{L}{2\theta_K^2}.
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

In particular,

```{math}
f(x^K) - f(x^\star) = O\!\left(\frac{1}{\theta_K^2}\right) = O\!\left(\frac{1}{K^2}\right).
```

Sweeping over {math}`K \in \llbracket 1, 100\rrbracket` gives the plot below, with
the theoretical bound in black and AutoLyap certificates as blue dots.

```{image} _static/optimized_gradient_method_c_vs_K_loglog.svg
:alt: Optimized-gradient c_K versus K in log-log scale with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## What to inspect

- `success`: whether the selected backend found a feasible certificate.
- `rho` (iteration-independent) or `c_K` (iteration-dependent): the main scalar output.
- `certificate`: Matrices and vectors that parameterize the Lyapunov certificate.

## Verbosity diagnostics

All three SDP entry points support a `verbosity` argument:

- {py:meth}`IterationIndependent.search_lyapunov <autolyap.IterationIndependent.search_lyapunov>`
- {py:meth}`IterationIndependent.LinearConvergence.bisection_search_rho <autolyap.IterationIndependent.LinearConvergence.bisection_search_rho>`
- {py:meth}`IterationDependent.search_lyapunov <autolyap.IterationDependent.search_lyapunov>`

Set:

- `verbosity=0` for silent mode (default).
- `verbosity=1` for concise diagnostic summaries.
- `verbosity=2` for detailed per-constraint/per-iteration diagnostics.

The diagnostic summary reports:

- nonnegativity checks on constrained scalars,
- PSD checks via minimum eigenvalues of constrained matrices,
- equality-constraint residuals (`max_abs_residual` and `l2_residual`).

Example:

```python
result = IterationIndependent.search_lyapunov(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    rho=1.0,
    solver_options=solver_options,
    verbosity=1,  # or 2 for detailed diagnostics
)
```

## Next

- For more worked walkthroughs, see {doc}`examples`.
- For the full API, see {doc}`api_reference`.
- For full mathematical context, see the companion paper {cite}`quick-upadhyaya2025autolyap`.

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: quick-
```
