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

To enable the MOSEK backend, install the optional extra:

```bash
pip install "autolyap[mosek]"
```

AutoLyap core dependencies:

* [NumPy](https://numpy.org/)
* [CVXPY](https://www.cvxpy.org/) (no MOSEK license required)

Optional backend dependency:

* [MOSEK](https://www.mosek.com/) (for `backend="mosek_fusion"`; academic license available)

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
