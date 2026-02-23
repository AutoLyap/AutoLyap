# Quick start

This page provides two end-to-end examples:

1. Iteration-independent analysis with a bisection search on `rho`.
2. Iteration-dependent analysis with chained Lyapunov inequalities.

Both examples follow the same modeling pattern and differ mainly in the
Lyapunov construction and target metric.

## Setup

For a high-level overview and navigation links, see {doc}`Home <index>`.

Install AutoLyap:

```bash
pip install autolyap
```

If you plan to use the MOSEK backend, install the optional MOSEK extra:

```bash
pip install "autolyap[mosek]"
```

Backend notes:

- `backend="mosek_fusion"` is recommended when a MOSEK license is available.
- `backend="cvxpy"` is license-free.
- MOSEK offers free academic licenses: [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/).
- If MOSEK is not licensed in your environment, run the same examples with
  `backend="cvxpy"` (for example, with `cvxpy_solver="CLARABEL"`).
- The examples below default to `backend="mosek_fusion"` and include a CVXPY
  fallback line.

## Workflow at a glance

Most analyses follow four steps:

1. Build an {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>` from {doc}`function classes <function_classes>` and {doc}`operator classes <operator_classes>`.
2. Pick an algorithm or your own subclass of {py:class}`Algorithm <autolyap.algorithms.Algorithm>`.
3. Select Lyapunov targets with helper constructors.
4. Solve the SDP and inspect `result["status"]`/`result["solve_status"]`,
   the scalar (`rho` or `c_K` when feasible), and `result["certificate"]`.

## Iteration-independent example: The gradient method

### Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is {math}`\mu`-strongly convex and
{math}`L`-smooth, with {math}`0 < \mu < L`.

For an initial point {math}`x^0 \in \calH` and step size
{math}`0 < \gamma < 2/L`, the gradient update is

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = x^k - \gamma \nabla f(x^k).
```

In this example, we search for the smallest contraction factor
{math}`\rho\in[0,1)` provable using AutoLyap such that

```{math}
\|x^k - x^\star\|^2 = O(\rho^k) \quad \textup{ as } \quad k\to\infty,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

### Model the problem in AutoLyap and search for the smallest rho

- {math}`f` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`GradientMethod <autolyap.algorithms.GradientMethod>`.
- Distance-to-solution parameters are obtained with
  {py:meth}`IterationIndependent.LinearConvergence.get_parameters_distance_to_solution <autolyap.IterationIndependent.LinearConvergence.get_parameters_distance_to_solution>`.
- The contraction factor is searched with
  {py:meth}`IterationIndependent.LinearConvergence.bisection_search_rho <autolyap.IterationIndependent.LinearConvergence.bisection_search_rho>`.

### Run the iteration-independent analysis

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
solver_options = SolverOptions(backend="mosek_fusion")  # requires `autolyap[mosek]`

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

rho = result["rho"]
certificate = result["certificate"]

rho_theory = max(gamma * L - 1.0, 1.0 - gamma * mu) ** 2
print(f"rho (AutoLyap): {rho:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

The computed value `rho (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical rate expression for gradient methods; see
{cite}`quick-Polyak1963GradientUSSR`:

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

### Problem setup

For background on the optimized gradient method, see
{cite}`quick-kim2015optimizedfirstorder`.

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is convex and {math}`L`-smooth with
{math}`L>0`.

For initial points {math}`x^0, y^0 \in \calH` and iteration budget
{math}`K \in \mathbb{N}`, the optimized gradient method updates as

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

### Model the problem in AutoLyap and search for the smallest c_K

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

### Run the iteration-dependent analysis

```python
from autolyap.algorithms import OptimizedGradientMethod
from autolyap import SolverOptions
from autolyap.problemclass import InclusionProblem, SmoothConvex
from autolyap.iteration_dependent import IterationDependent

L = 1.0
K = 5

problem = InclusionProblem([SmoothConvex(L)])
algorithm = OptimizedGradientMethod(L=L, K=K)
solver_options = SolverOptions(backend="mosek_fusion")  # requires `autolyap[mosek]`

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
certificate = result["certificate"]

Q_sequence = certificate["Q_sequence"]  # [Q_0, Q_1, ..., Q_K]
q_sequence = certificate["q_sequence"]  # [q_0, q_1, ..., q_K] or None

theta_K = algorithm.compute_theta(K, K)
c_K_theory = L / (2.0 * theta_K ** 2)

print(f"c_K (AutoLyap): {c_K:.6e}")
print(f"c_K (theory):   {c_K_theory:.6e}")
```

The computed value `c_K (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical horizon-`K` expression:

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

- `status`: one of `feasible`, `infeasible`, or `not_solved`.
- `solve_status`: backend-specific raw solver status (or `solver_error` / `optimize_error`).
- `rho` (iteration-independent) or `c_K` (iteration-dependent): the main scalar output.
- `certificate`: Matrices and vectors that parameterize the Lyapunov certificate.

## Verbosity diagnostics

All three SDP entry points support a `verbosity` argument:

- {py:meth}`IterationIndependent.search_lyapunov <autolyap.IterationIndependent.search_lyapunov>`
- {py:meth}`IterationIndependent.LinearConvergence.bisection_search_rho <autolyap.IterationIndependent.LinearConvergence.bisection_search_rho>`
- {py:meth}`IterationDependent.search_lyapunov <autolyap.IterationDependent.search_lyapunov>`

Set:

- `verbosity=1` for concise diagnostic summaries (default).
- `verbosity=0` for silent mode.
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

- For theoretical foundations, see {doc}`theory`.
- For more worked walkthroughs, see {doc}`examples`.
- For the full API, see {doc}`api_reference`.

## References

```{bibliography}
:filter: docname in docnames
:keyprefix: quick-
```
