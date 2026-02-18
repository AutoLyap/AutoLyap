---
tocdepth: 1
---

```{include} ../../README.md
:end-before: "## Documentation"
```

## Quick start

For installation instructions and first end-to-end workflows, see
{doc}`Quick start <quick_start>`.

## Cite this project

If AutoLyap contributes to your research or software, please cite
{cite}`index-upadhyaya2025autolyap`.

```bibtex
@misc{upadhyaya2025autolyap,
  author = {Upadhyaya, Manu and Taylor, Adrien B. and Banert, Sebastian and Giselsson, Pontus},
  title = {{AutoLyap}: {A} {P}ython package for computer-assisted Lyapunov analyses for first-order methods},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2506.24076},
  primaryClass = {math.OC},
}
```

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
Theory <theory>
examples
api_reference
contributing
whats_new
```
