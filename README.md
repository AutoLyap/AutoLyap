# AutoLyap

A Python package for automated Lyapunov-based convergence analysis of first-order optimization and inclusion methods.

---

## Overview

AutoLyap streamlines the process of constructing and verifying Lyapunov analyses by formulating them as semidefinite programs (SDPs). It supports a broad class of structured optimization and inclusion problems, providing computer-assisted proofs of linear or sublinear convergence rates for many well‑known algorithms.

A typical workflow:
1. Choose the class of optimization/inclusion problems.
2. Choose the first-order method to analyze.
3. Choose the type of Lyapunov analysis to search for or verify (which implies a convergence or performance conclusion).

AutoLyap builds the SDP and solves it with MOSEK Fusion by default. It can also
solve through CVXPY when requested.

## Documentation

User guide and API reference:
➡️  [https://autolyap.github.io](https://autolyap.github.io/)

## Installation

```bash
pip install autolyap
```

AutoLyap depends on:

* [NumPy](https://numpy.org/) 
* [MOSEK](https://docs.mosek.com/latest/install/installation.html) (academic license available)
* [CVXPY](https://www.cvxpy.org/)

## Solver backends

Detailed backend selection, backend-specific parameter configuration, no-license
CVXPY usage, and troubleshooting are documented in:

- [Solver backends](https://autolyap.github.io/solver_backends.html)

## Companion paper

The complete mathematical development and examples to get started are available in the companion paper on [arXiv](https://arxiv.org/abs/2506.24076).

## Source code

GitHub repository: [AutoLyap/AutoLyap](https://github.com/AutoLyap/AutoLyap)

## Development install & tests

If you want to work on the source or run the test suite:

```bash
git clone https://github.com/AutoLyap/AutoLyap.git
cd AutoLyap
python -m pip install -e '.[test]'
python -m pytest
```

MOSEK-backed tests will skip automatically if MOSEK or a valid license is not available.
CVXPY backend smoke tests will skip if CVXPY (or a supported CVXPY SDP solver)
is unavailable. Pytest is configured to show skip reasons.

## Troubleshooting

- `MOSEK license error`:
  Set `solver_options=SolverOptions(backend="cvxpy", cvxpy_solver="CLARABEL")`
  (or `"SCS"`) to run without MOSEK licensing.
- `CVXPY solver not installed`:
  Install a CVXPY backend solver (for example: `python -m pip install cvxpy clarabel`)
  and retry with `backend="cvxpy"`.

## Build docs

To build the Sphinx documentation locally:

```bash
make -C docs deps
make -C docs html
```

The generated site is written to `docs/build/html/` (open `docs/build/html/index.html`).

## Other computer-assisted methodologies

[PEPit](https://pepit.readthedocs.io) is a computer-assisted PEP framework that targets worst-case analyses of first-order methods through SDP formulations. AutoLyap is complementary: it focuses on Lyapunov analyses and automates the corresponding SDP formulations. In practice, PEPit is a strong choice for tight bounds, while AutoLyap is tailored to Lyapunov-based proofs and scalable analysis patterns.
