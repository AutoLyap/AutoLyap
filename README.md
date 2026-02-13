# AutoLyap

A Python package for automated Lyapunov-based convergence analyses of first-order optimization and inclusion methods.

---

## Overview

AutoLyap streamlines the process of constructing and verifying Lyapunov analyses by formulating them as semidefinite programs (SDPs). It supports a broad class of structured optimization and inclusion problems, providing computer-assisted proofs of linear or sublinear convergence rates for many well‑known algorithms.

A typical workflow:
1. Choose the class of optimization/inclusion problems.
2. Choose the first-order method to analyze.
3. Choose the type of Lyapunov analysis to search for or verify (which implies a convergence or performance conclusion).

AutoLyap builds the underlying SDP and solves it through configurable backend
solvers.

## Documentation

User guide and API reference: [https://autolyap.github.io](https://autolyap.github.io/)

## Note about the docs

To build the documentation locally:

```bash
make -C docs deps
make -C docs html
```

The generated site is written to `docs/build/html/` (open `docs/build/html/index.html`).
