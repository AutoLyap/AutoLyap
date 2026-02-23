# The accelerated proximal point method

## Problem setup

Consider the monotone inclusion

```{math}
\text{find } x \in \calH \text{ such that } 0 \in G(x),
```

where {math}`G:\calH\rightrightarrows\calH` is maximally monotone.

For initial points {math}`x^0,y^0,y^{-1}\in\calH`, step size
{math}`\gamma\in\reals_{++}`, and
{math}`\lambda_k = k/(k+2)`, the accelerated proximal point method
{cite}`kim2021acceleratedproximalpoint` is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
x^{k+1} &= J_{\gamma G}(y^k), \\
y^{k+1} &= x^{k+1} + \lambda_k(x^{k+1} - x^k) - \lambda_k(x^k - y^{k-1}).
\end{aligned}
\right.
```

In this example, we use AutoLyap to compute the smallest certificate constant
{math}`c_K` such that

```{math}
\|x^{K+1} - y^K\|^2 \le c_K \|y^0 - y^\star\|^2,
```

where {math}`K \in \mathbb{N}` is the iteration budget (horizon), and
{math}`y^\star \in \zer G`.

## Model the problem in AutoLyap and search for the smallest c_K

- {math}`G` is modeled by
  {py:class}`MaximallyMonotone <autolyap.problemclass.MaximallyMonotone>`.
- The monotone inclusion is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`AcceleratedProximalPoint <autolyap.algorithms.AcceleratedProximalPoint>`.
- Initial-horizon parameters
  {math}`\|y^0-y^\star\|^2` are obtained with
  {py:meth}`IterationDependent.get_parameters_state_component_distance_to_solution <autolyap.IterationDependent.get_parameters_state_component_distance_to_solution>`.
- Final-horizon parameters
  {math}`\|x^{K+1}-y^K\|^2` are obtained with
  {py:meth}`IterationDependent.get_parameters_state_component_cross_iteration_difference <autolyap.IterationDependent.get_parameters_state_component_cross_iteration_difference>`.
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
from autolyap.algorithms import AcceleratedProximalPoint
from autolyap.problemclass import InclusionProblem, MaximallyMonotone

gamma = 1.0
K = 10

problem = InclusionProblem([MaximallyMonotone()])
algorithm = AcceleratedProximalPoint(gamma=gamma, type="operator")
solver_options = SolverOptions(backend="mosek_fusion")

# V(0) = ||y^0 - y^*||^2. In APP state x^k = (x^k, y^k, y^{k-1}),
# y^k is state component ell=2 (1-indexed).
Q_0 = IterationDependent.get_parameters_state_component_distance_to_solution(
    algorithm,
    k=0,
    ell=2,
)

# V(K) = ||x^{K+1} - y^K||^2 = ||x_1^{K+1} - x_2^K||^2
# with 1-indexed state components: ell=1 and ell_prime=2.
Q_K = IterationDependent.get_parameters_state_component_cross_iteration_difference(
    algorithm,
    k=K,
    ell=1,
    ell_prime=2,
)

result = IterationDependent.search_lyapunov(
    problem,
    algorithm,
    K,
    Q_0,
    Q_K,
    solver_options=solver_options,
)
if result["status"] != "feasible":
    raise RuntimeError("No feasible chained Lyapunov certificate for APP.")

c_K_autolyap = result["c_K"]
c_K_kim = 1.0 / (K + 1.0) ** 2

print(f"APP c_K (AutoLyap): {c_K_autolyap:.6e}")
print(f"APP c_K (Kim):      {c_K_kim:.6e}")
```

The computed value `APP c_K (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical horizon-`K` expression from
{cite}`kim2021acceleratedproximalpoint{Theorem 4.1}`, i.e.,

```{math}
\|x^{K+1} - y^K\|^2 \le \frac{1}{(K+1)^2}\|y^0 - y^\star\|^2.
```

Sweeping over {math}`K \in \llbracket 1, 100\rrbracket` gives the plot
below, with Kim's theoretical constant from
{cite}`kim2021acceleratedproximalpoint{Theorem 4.1}`
{math}`c_K^{\mathrm{Kim}} = \frac{1}{(K+1)^2}` as a line and AutoLyap
certificates as blue dots.

```{image} ../_static/accelerated_proximal_point_method_c_vs_K_loglog.svg
:alt: Accelerated-proximal-point c_K versus K in log-log scale with Kim's bound and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
