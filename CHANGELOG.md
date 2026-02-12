# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-11

### Added

- Top-level `SolverOptions` support for explicit backend configuration in
  analysis calls.
- New quick-start documentation and worked examples for proximal methods:
  proximal gradient and proximal point.
- Runtime diagnostic summaries for Lyapunov solves, including checks for
  nonnegativity-constrained scalars, PSD-constrained matrices (minimum-eigenvalue
  checks), and equality-constraint residuals.
- `verbosity` controls for iteration-independent and iteration-dependent
  Lyapunov verification/search APIs.
- Explicit backend test coverage for MOSEK and CVXPY execution paths.

### Changed

- Refactored problem-class internals into focused modules (`base`,
  `functions`, `operators`, `indices`, `inclusion_problem`) with stricter
  validation and clearer error messages.
- Updated Lyapunov verification/search routines to return certificates in solver
  results.
- Improved documentation structure and navigation (quick start, API layout,
  release notes organization).
- Expanded and reorganized test coverage across algorithms, convergence checks,
  backends, and problem-class validation.
- Added CI workflow coverage for automated testing.

### Breaking

- Renamed interpolation index tokens:
  - `i<j` -> `r1<r2`
  - `i!=j` -> `r1!=r2`
  - `i` -> `r1`
  - `i!=star` -> `r1!=star`
