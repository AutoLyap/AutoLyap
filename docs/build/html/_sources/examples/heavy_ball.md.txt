# The heavy-ball method

## Problem setup

We consider the unconstrained minimization of a convex and {math}`L`-smooth
function {math}`f : \calH \to \reals`, with {math}`L > 0`.

Given initial points {math}`x^{-1}, x^0 \in \calH`, momentum
{math}`\delta \in \reals`, and step size {math}`\gamma \in \reals_{++}`,
the heavy-ball iteration is

```{math}
(\forall k \in \naturals)\quad
x^{k+1} = x^k - \gamma \nabla f(x^k) + \delta(x^k - x^{k-1}).
```

This method was introduced in {cite}`Polyak1964HeavyBall`.

In this example, we certify function-value suboptimality in the sublinear
setting ({math}`\rho = 1`) while enforcing condition (C4), yielding an
{math}`o(1/k)` conclusion.

## Run the iteration-independent analysis with C4

```python
from autolyap import IterationIndependent
from autolyap.algorithms import HeavyBallMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex

L = 1.0
gamma = 1.0
delta = 0.4

problem = InclusionProblem([SmoothConvex(L)])
algorithm = HeavyBallMethod(gamma=gamma, delta=delta)

# V(P,p,k) = 0 and R(T,t,k) = function-value suboptimality
P, p, T, t = IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality(
    algorithm,
    tau=0,
)

result = IterationIndependent.verify_iteration_independent_Lyapunov(
    problem,
    algorithm,
    P,
    T,
    p=p,
    t=t,
    rho=1.0,
    remove_C4=False,
)

if not result["success"]:
    raise RuntimeError("No feasible Lyapunov certificate for this (gamma, delta) pair.")

print("Feasible certificate found with C4 enabled.")
```

When the certificate is feasible, the certified function-value convergence is

```{math}
f(x^k) - f(x^\star) = o\!\left(\frac{1}{k}\right),
```

where {math}`x^\star \in \Argmin_{x \in \calH} f(x)`.

Equivalently,

```{math}
\lim_{k \to \infty} k\bigl(f(x^k)-f(x^\star)\bigr)=0.
```

Sweeping over {math}`\gamma` and {math}`\delta` gives the plot below, where
each dot denotes a parameter pair for which the verification is successful.

```{image} ../_static/heavy_ball_smooth_convex.svg
:alt: Certified heavy-ball smooth-convex region in the (gamma, delta) plane.
:width: 100%
```

## References

```{bibliography}
:filter: docname in docnames
```
