# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Curated, user-facing summaries are available in
[`docs/source/release_notes/`](docs/source/release_notes/).

## [Unreleased]

## [0.2.0] - 2026-02-11

### Added

- Top-level `SolverOptions` support for explicit backend configuration in
  Lyapunov verification/search calls.
- CVXPY backend execution paths alongside MOSEK Fusion, with backend
  equivalence and integration coverage.
- Runtime diagnostic summaries for Lyapunov solves, including checks for
  nonnegativity-constrained scalars, PSD-constrained matrices
  (minimum-eigenvalue checks), and equality-constraint residuals.
- `verbosity` controls for iteration-independent and iteration-dependent
  Lyapunov verification/search APIs.
- New quick-start documentation and expanded worked examples for proximal,
  heavy-ball, Nesterov fast-gradient, and Douglas-Rachford analyses.
- CI workflows for tests, releases, CodeQL, and secret scanning.

### Changed

- Refactored problem-class internals into focused modules (`base`,
  `functions`, `operators`, `indices`, `inclusion_problem`) with stricter
  validation and clearer error messages.
- Updated Lyapunov verification/search routines to return certificates in
  solver results.
- Tightened algorithm constructor/setter validation and shared helper
  utilities for dimensions and matrix checks.
- Improved documentation structure and navigation (quick start, API layout,
  release notes organization).
- Expanded and reorganized test coverage across algorithms, convergence checks,
  backends, and problem-class validation.
- Updated release/contributing guidance and local docs-build instructions.

### Breaking changes

- Renamed interpolation index tokens:
  - `i<j` -> `r1<r2`
  - `i!=j` -> `r1!=r2`
  - `i` -> `r1`
  - `i!=star` -> `r1!=star`
- Updated public return shapes for Lyapunov APIs:
  - `IterationIndependent.verify_iteration_independent_Lyapunov` now returns
    `{"success", "rho", "certificate"}` (previously `bool`).
  - `IterationIndependent.LinearConvergence.bisection_search_rho` now returns
    `{"success", "rho", "certificate"}` (previously `float | None`).
  - `IterationDependent.verify_iteration_dependent_Lyapunov` now returns
    `{"success", "c_K", "certificate"}` (previously `(bool, c)`).

[Unreleased]: https://github.com/AutoLyap/AutoLyap/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AutoLyap/AutoLyap/releases/tag/v0.2.0
