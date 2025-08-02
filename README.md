# AutoLyap

A Python package for automated Lyapunov-based convergence analysis of first-order optimization and inclusion methods.

---

## Overview

AutoLyap streamlines the process of constructing and verifying Lyapunov analyses by formulating them as semidefinite programs (SDPs). It supports a broad class of structured optimization and inclusion problems, automating proofs of linear or sublinear convergence rates for many well‑known algorithms.

## Documentation

User guide and API reference:
➡️  [https://autolyap.github.io](https://autolyap.github.io/)

## Companion paper

The complete mathematical development and examples to get started are available in the companion paper on [arXiv](https://arxiv.org/abs/2506.24076).

## Installation

```bash
pip install autolyap
```

AutoLyap depends on:

* [NumPy](https://numpy.org/) 
* [MOSEK](https://www.mosek.com/) (academic license available)
