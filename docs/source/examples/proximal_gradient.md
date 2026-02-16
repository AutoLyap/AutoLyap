# Define your own algorithm: The proximal gradient method

This example is a step-by-step guide to define your own concrete algorithm with
{py:class}`Algorithm <autolyap.algorithms.Algorithm>`.

## Step 1: Problem statement

Let {math}`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space with canonical norm 
{math}`\|\cdot\|`.

Use the inclusion form

```{math}
\text{find } y \in \calH \text{ such that } 0 \in \sum_{i \in \IndexFunc} \partial f_i(y)
+ \sum_{i \in \IndexOp} G_i(y),
```

with two functional components and no operator component:

- {math}`f_1 = f`, where
  ```{math}
  f:\calH \to \mathbb{R}
  ```
  is {math}`L`-smooth and {math}`\mu`-strongly convex
  ({math}`0 < \mu < L < +\infty`).
- {math}`f_2 = g`, where
  ```{math}
  g:\calH \to \mathbb{R}\cup\{+\infty\}
  ```
  is proper, lower semicontinuous, convex, and
  ```{math}
  \partial g:\calH \rightrightarrows \calH.
  ```
- {math}`\IndexFunc = \{1,2\}` and {math}`\IndexOp = \emptyset`.

Equivalently,

```{math}
\text{find } y \in \calH \text{ such that } 0 \in \nabla f(y) + \partial g(y).
```

## Step 2: Standard form

For an initial point {math}`x^0 \in \calH` and step size
{math}`\gamma \in \reals_{++}` with {math}`0 < \gamma \le 2/L`,

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = \prox_{\gamma g}\!\left(x^k - \gamma \nabla f(x^k)\right).
```

Using the optimality condition of the proximal operator, this is equivalent to

```{math}
\frac{x^k - x^{k+1}}{\gamma} - \nabla f(x^k) \in \partial g(x^{k+1})
```

or 
```{math}
x^{k+1} \in x^k - \gamma \nabla f(x^k) - \gamma \partial g(x^{k+1}).
```

## Step 3: State-space representation

Match the base representation used by
{py:class}`Algorithm <autolyap.algorithms.Algorithm>`

```{math}
\begin{aligned}
\bx^{k+1} &= (A_k \kron \Id)\bx^k + (B_k \kron \Id)\bu^k,\\
\by^k &= (C_k \kron \Id)\bx^k + (D_k \kron \Id)\bu^k,\\
(\bu_i^k)_{i \in \IndexFunc} &\in \prod_{i \in \IndexFunc} \boldsymbol{\partial}\bfcn_i(\by_i^k)
\end{aligned}
```

with

```{math}
\begin{aligned}
\bx^k &= x^k,\\
\bu^k &= \left(\nabla f(x^k),\, \frac{x^k - x^{k+1}}{\gamma} - \nabla f(x^k)\right),\\
\by^k &= (x^k, x^{k+1}),\\
\boldsymbol{\partial}\bfcn_1 &: \calH \rightrightarrows \calH : y \mapsto  \partial f(y) = \{\nabla f(y)\},\\
\boldsymbol{\partial}\bfcn_2 &: \calH \rightrightarrows \calH : y \mapsto  \partial g(y).
\end{aligned}
```

and the matrices

```{math}
\begin{aligned}
A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
B_k &= \begin{bmatrix} -\gamma & -\gamma \end{bmatrix}, \\
C_k &=
\begin{bmatrix}
1 \\
1
\end{bmatrix}, &
D_k &=
\begin{bmatrix}
0 & 0 \\
-\gamma & -\gamma
\end{bmatrix}.
\end{aligned}
```

## Step 4: Implement the custom algorithm

```python
import numpy as np
from typing import Tuple

from autolyap.algorithms import Algorithm

class ProximalGradientMethod(Algorithm):
    def __init__(self, gamma):
        super().__init__(n=1, m=2, m_bar_is=[1, 1], I_func=[1, 2], I_op=[])
        self.set_gamma(gamma)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1.0]])
        B = np.array([[-self.gamma, -self.gamma]])
        C = np.array([[1.0], [1.0]])
        D = np.array([
            [0.0, 0.0],
            [-self.gamma, -self.gamma],
        ])
        return A, B, C, D
```

## Step 5: Build {py:class}`InclusionProblem <autolyap.problemclass.InclusionProblem>` and run the analysis

This example uses the MOSEK Fusion backend (`backend="mosek_fusion"`).
Install the optional MOSEK dependency first:

```bash
pip install "autolyap[mosek]"
```

```python
from autolyap import IterationIndependent, SolverOptions
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex


def validate_parameters(mu: float, L: float, gamma: float) -> None:
    if not (0.0 < mu < L):
        raise ValueError(
            f"Invalid parameters: require 0 < mu < L. Got mu={mu}, L={L}."
        )

    gamma_max = 2.0 / L
    if not (0.0 < gamma <= gamma_max):
        raise ValueError(
            f"Invalid parameters: require 0 < gamma <= 2/L. Got gamma={gamma}, 2/L={gamma_max}."
        )


mu = 1.0
L = 4.0
gamma = 2.0 / (L + mu)
validate_parameters(mu=mu, L=L, gamma=gamma)

problem = InclusionProblem([
    SmoothStronglyConvex(mu, L),    # component i=1: f
    Convex(),                       # component i=2: g
])
algorithm = ProximalGradientMethod(gamma=gamma)
solver_options = SolverOptions(backend="mosek_fusion")

P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
    algorithm,
    i=1,
    j=1,
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
rho_taylor = max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2

print(f"rho (AutoLyap):  {rho_autolyap:.8f}")
print(f"rho (theory):    {rho_taylor:.8f}")
```

The computed value `rho (AutoLyap)` matches the closed-form theoretical rate
expression from Theorem 2.1 in {cite}`taylor2018proximal`, i.e.,

```{math}
\|x^k - x^\star\|^2 = O(\rho^k), \qquad
\rho = \max\{|1-\gamma L|,\;|1-\gamma\mu|\}^2,
```

where {math}`x^\star \in \Argmin_{x \in \calH} \bigl(f(x)+g(x)\bigr)`.

Equivalently,

```{math}
\|x^k - x^\star\| = O\!\left(\max\{|1-\gamma L|,\;|1-\gamma\mu|\}^k\right).
```

Sweeping over 100 values of {math}`\gamma` on {math}`0 < \gamma \le 2/L` gives
the plot below, with the theoretical rate in black and AutoLyap certificates
as blue dots.

```{image} ../_static/proximal_gradient_rho_vs_gamma.svg
:alt: Proximal-gradient rho versus gamma with theoretical line and AutoLyap points.
:align: center
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
