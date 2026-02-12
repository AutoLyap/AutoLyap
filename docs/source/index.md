---
tocdepth: 1
---

```{include} ../../README.md
:end-before: "## Documentation"
```

## Installation

```bash
pip install autolyap
```

AutoLyap depends on:

* [NumPy](https://numpy.org/)
* [MOSEK](https://www.mosek.com/) (academic license available)
* [CVXPY](https://www.cvxpy.org/) (no MOSEK license required)

## Companion paper

For full mathematical context, see the companion paper {cite}`index-upadhyaya2025autolyap`.

```{bibliography}
:filter: docname in docnames
:keyprefix: index-
```

## Other computer-assisted methodologies

[PEPit](https://pepit.readthedocs.io) is a computer-assisted performance estimation framework that targets worst-case analyses of first-order methods through SDP formulations. AutoLyap is complementary: it focuses on Lyapunov analyses and automates the corresponding SDP formulations. In practice, PEPit is a strong choice for tight bounds, while AutoLyap is tailored to Lyapunov-based proofs and scalable analysis patterns.

```{toctree}
:maxdepth: 1
:hidden:

Home <self>
quick_start
examples
api_reference
contributing
whats_new
```
