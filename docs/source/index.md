---
tocdepth: 1
---

<!-- SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

```{include} ../../README.md
:end-before: "## Documentation"
```

## Quick start

For installation instructions and first end-to-end workflows, see
{doc}`Quick start <quick_start>`.

## Cite this project

If AutoLyap contributes to your research or software, please cite
{cite}`index-upadhyaya2026autolyap`.

```bibtex
@misc{upadhyaya2026autolyap,
  author = {Upadhyaya, Manu and Das Gupta, Shuvomoy and Taylor, Adrien B. and Banert, Sebastian and Giselsson, Pontus},
  title = {The {AutoLyap} software suite for computer-assisted {L}yapunov analyses of first-order methods},
  year = {2026},
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

## Contributors

- [Manu Upadhyaya](https://manuupadhyaya.github.io/): Creator and maintainer.
- [Shuvomoy Das Gupta](https://shuvomoy.github.io/): Creator; maintainer of [AutoLyap.jl](https://github.com/AutoLyap/AutoLyap.jl) (Julia version of AutoLyap).
- [Adrien B. Taylor](https://adrientaylor.github.io/): Creator.
- [Sebastian Banert](https://github.com/sbanert): Creator.
- [Pontus Giselsson](https://portal.research.lu.se/en/persons/pontus-giselsson/): Creator.

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
