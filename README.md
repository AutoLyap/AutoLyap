# AutoLyap

A Python package for automated Lyapunov-based convergence analyses of first-order optimization and inclusion methods.

[![PyPI version](https://img.shields.io/pypi/v/autolyap?style=flat&logo=pypi&logoColor=white&label=PyPI&labelColor=111111&color=5fa8e8)](https://pypi.org/project/autolyap/)
[![PyPI downloads](https://img.shields.io/pepy/dt/autolyap?style=flat&label=downloads&labelColor=111111&color=5fa8e8)](https://pepy.tech/projects/autolyap)
[![GitHub stars](https://img.shields.io/github/stars/AutoLyap/AutoLyap?style=flat&logo=github&logoColor=white&label=stars&labelColor=111111&color=5fa8e8)](https://github.com/AutoLyap/AutoLyap/stargazers)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2506.24076-5fa8e8?style=flat&logo=arxiv&logoColor=white&labelColor=111111)](https://arxiv.org/abs/2506.24076)

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

- User docs: [https://autolyap.github.io](https://autolyap.github.io/)
- Contributing guide: [https://autolyap.github.io/contributing/](https://autolyap.github.io/contributing/)
- Developer commands (internal): [`DEVELOPER_COMMANDS.md`](DEVELOPER_COMMANDS.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Release process (maintainers): [`RELEASING.md`](RELEASING.md)
- License: [`LICENSE`](LICENSE)

## Cite this project

If AutoLyap contributes to your research or software, please cite:

- Upadhyaya, Manu; Das Gupta, Shuvomoy; Taylor, Adrien B.; Banert, Sebastian; Giselsson, Pontus (2026). *The AutoLyap software suite for computer-assisted Lyapunov analyses of first-order methods*. arXiv:2506.24076.

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
