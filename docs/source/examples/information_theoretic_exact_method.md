# The information-theoretic exact method

## Problem setup

Consider the unconstrained minimization problem

```{math}
\minimize_{x \in \calH} f(x),
```

where {math}`f : \calH \to \reals` is {math}`\mu`-strongly convex and
{math}`L`-smooth with {math}`0 < \mu < L`.

For initial points {math}`x^0, z^0 \in \calH`, set
{math}`q=\mu/L`, {math}`y^{-1}=x^0=z^0`, and {math}`\tilde{A}_0=0`.
For each {math}`k \in \naturals`, define

```{math}
\begin{aligned}
\tilde{A}_{k+1}
&=
\frac{(1+q)\tilde{A}_k + 2\left(1 + \sqrt{(1+\tilde{A}_k)(1+q\tilde{A}_k)}\right)}
{(1-q)^2},
\\
\beta_k
&=
\frac{\tilde{A}_k}{(1-q)\tilde{A}_{k+1}},
\\
\delta_k
&=
\frac{(1-q)^2\tilde{A}_{k+1}-(1+q)\tilde{A}_k}{2(1+q+q\tilde{A}_k)}.
\end{aligned}
```

The information-theoretic exact method (ITEM)
{cite}`taylor2023optimalgradientmethod` is

```{math}
(\forall k \in \naturals)\quad
\left[
\begin{aligned}
y^k &= (1-\beta_k)z^k + \beta_k x^k, \\
x^{k+1} &= y^k - \frac{1}{L}\nabla f(y^k), \\
z^{k+1} &= (1-q\delta_k)z^k + q\delta_k y^k - \frac{\delta_k}{L}\nabla f(y^k).
\end{aligned}
\right.
```

In this example, we search for the smallest AutoLyap certificate constant
{math}`c_K` such that

```{math}
\|z^K-x^\star\|^2 \le c_K\|z^0-x^\star\|^2,
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

## Model the problem in AutoLyap and search for the smallest c_K

- {math}`f` is modeled by
  {py:class}`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`.
- The optimization problem is represented by
  {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>`.
- The update rule is represented by
  {py:class}`ITEM <autolyap.algorithms.ITEM>`.
- Initial and final distance-to-solution parameters
  {math}`\|z^k-x^\star\|^2` are obtained with
  {py:meth}`IterationDependent.get_parameters_state_component_distance_to_solution <autolyap.IterationDependent.get_parameters_state_component_distance_to_solution>`
  using `ell=2` (ITEM state is {math}`(x^k,z^k)`).
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
from autolyap.algorithms import ITEM
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex

mu = 1.0
L = 200.0
K = 10
q = mu / L

problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L)])
algorithm = ITEM(mu=mu, L=L)
solver_options = SolverOptions(backend="mosek_fusion")
# License-free option:
# solver_options = SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")

Q_0, q_0 = IterationDependent.get_parameters_state_component_distance_to_solution(
    algorithm,
    k=0,
    ell=2,
)
Q_K, q_K = IterationDependent.get_parameters_state_component_distance_to_solution(
    algorithm,
    k=K,
    ell=2,
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
    raise RuntimeError("No feasible chained Lyapunov certificate for ITEM.")

c_K_autolyap = result["c_K"]
A_tilde_K = algorithm.get_A(K)
c_K_theory = 1.0 / (1.0 + q * A_tilde_K)

print(f"ITEM c_K (AutoLyap): {c_K_autolyap:.6e}")
print(f"ITEM c_K (theory):   {c_K_theory:.6e}")
```

The computed value `ITEM c_K (AutoLyap)` matches (up to solver numerical
tolerances) the theoretical horizon-`K` expression from
{cite}`taylor2023optimalgradientmethod{Theorem 3}`:

```{math}
\|z^K-x^\star\|^2 \le c_K\|z^0-x^\star\|^2,
\qquad
c_K = \frac{1}{1+q\tilde{A}_K}.
```

Sweeping over {math}`K \in \llbracket 1, 100\rrbracket` gives the log-log plot
below, with the theoretical bound in black and AutoLyap certificates as blue
dots.

```{image} ../_static/information_theoretic_exact_method_c_vs_K_loglog.svg
:alt: ITEM c_K versus K in log-log scale with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
