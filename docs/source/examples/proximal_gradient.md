# Define your own algorithm: The proximal gradient method

This example is a step-by-step guide to define your own concrete algorithm with
{py:class}`autolyap.algorithms.Algorithm`.

## Step 1: Problem statement

Let {math}`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space with norm canonical 
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

Using the proximal optimality condition, this is equivalent to

```{math}
(\forall k \in \naturals)\quad
\frac{x^k - x^{k+1}}{\gamma} - \nabla f(x^k) \in \partial g(x^{k+1}).
```

## Step 3: State-space representation

Match the base representation

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
\by^k &= (y_1^k, y_2^k) = (x^k, x^{k+1}),\\
\boldsymbol{\partial}\bfcn_1 &: \calH \rightrightarrows \calH : y \mapsto  \partial f (x) = \{\nabla f(y)\},\\
\boldsymbol{\partial}\bfcn_2 &: \calH \rightrightarrows \calH : y \mapsto  \partial g(y).
\end{aligned}
```

and the matrices in {py:meth}`autolyap.algorithms.Algorithm.get_ABCD` are

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

## Step 5: Build `InclusionProblem` and run the analysis

```python
from autolyap import IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem, SmoothStronglyConvex

mu = 1.0
L = 4.0
gamma = 2.0 / (L + mu)

problem = InclusionProblem([
    SmoothStronglyConvex(mu, L),    # component i=1: f
    Convex(),                       # component i=2: g
])
algorithm = ProximalGradientMethod(gamma=gamma)

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
)

if not result["success"]:
    raise RuntimeError("No feasible Lyapunov certificate in the requested rho interval.")

rho_autolyap = result["rho"]
rho_theory = max(abs(1.0 - L * gamma), abs(1.0 - mu * gamma)) ** 2

print(f"rho (AutoLyap): {rho_autolyap:.8f}")
print(f"rho (theory):   {rho_theory:.8f}")
```

## Notes

- The theoretical benchmark above follows the exact worst-case rate in
  Theorem 2.1 of Taylor, Hendrickx, and Glineur
  ([Journal of Optimization Theory and Applications, 2018](https://link.springer.com/article/10.1007/s10957-018-1298-1)).
