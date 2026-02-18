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

- User docs: [https://autolyap.github.io](https://autolyap.github.io/)
- Contributing guide: [https://autolyap.github.io/contributing/](https://autolyap.github.io/contributing/)
- Developer commands (internal): [`DEVELOPER_COMMANDS.md`](DEVELOPER_COMMANDS.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Release process (maintainers): [`RELEASING.md`](RELEASING.md)
- License: [`LICENSE`](LICENSE)

## Cite this project

If AutoLyap contributes to your research or software, please cite:

- Upadhyaya, Manu; Taylor, Adrien B.; Banert, Sebastian; Giselsson, Pontus (2025). *AutoLyap: A Python package for computer-assisted Lyapunov analyses for first-order methods*. arXiv:2506.24076.

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
