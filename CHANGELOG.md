# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Curated, user-facing summaries are available in
[`docs/source/release_notes/`](docs/source/release_notes/).

## [v0.2.0] - 2026-03-02

### Added

- Top-level `SolverOptions` API support for explicit backend configuration in
  Lyapunov search calls.
- First-class CVXPY backend support alongside MOSEK Fusion, including solver
  profiles for `CLARABEL`, `SCS`, `MOSEK`, `SDPA`, `SDPA` multiprecision, and
  `COPT`.
- Protocol-based backend typing via `autolyap/utils/backend_types.py`, shared
  across CVXPY and MOSEK execution paths.
- Certificate returns from Lyapunov SDP routines (iteration-independent,
  iteration-dependent, and bisection search outputs).
- Runtime diagnostic summaries for Lyapunov solves, including checks for
  nonnegativity-constrained scalars, PSD-constrained matrices
  (minimum-eigenvalue checks), and equality-constraint residuals.
- `verbosity` controls for iteration-independent and iteration-dependent
  Lyapunov search APIs.
- A large documentation expansion: new quick start, solver-backend guidance,
  release notes scaffolding, and a dedicated theory section (notation, problem
  classes, algorithm representation, interpolation conditions, and Lyapunov
  analyses).
- New worked examples and generated assets for proximal gradient, proximal
  point, heavy-ball, Nesterov momentum, Nesterov fast gradient, optimized
  gradient method, accelerated proximal point, ITEM, Chambolle-Pock, Davis-Yin,
  Malitsky-Tam FRB, and multiple Douglas-Rachford regimes.
- Citation metadata and tooling: `CITATION.cff`, richer bibliography coverage,
  and `scripts/sync_citation_version.py` with `make sync-citation` and
  `make check-citation`.
- Project/release metadata and local tooling files: `VERSION`,
  `DEVELOPER_COMMANDS.md`, `RELEASING.md`, root `Makefile`, `pytest.ini`, and
  local check helpers.
- Reorganized and expanded test suites by area
  (`algorithm/`, `backend/`, `convergence/`, `lyapunov/`, `problemclass/`,
  `solver/`, `shared/`) with stronger backend equivalence and convergence
  coverage.
- New/expanded CI automation for release publishing, docs publishing, CodeQL,
  gitleaks, docs-token validation, and stricter MOSEK validation paths.

### Changed

- Refactored problem-class internals into focused modules (`base`,
  `functions`, `operators`, `indices`, `inclusion_problem`) with stricter
  validation and clearer error messages.
- Tightened Lyapunov result-status semantics across backends:
  - `status="feasible"` only when a certificate is returned.
  - `status="infeasible"` for genuine infeasibility.
  - `status="not_solved"` for solver/interface failures or indeterminate
    statuses.
- Updated bisection search behavior to treat intermediate
  `status="not_solved"` checks conservatively during interval updates, while
  preserving terminal status reporting.
- Set default Lyapunov-search verbosity to `verbosity=1` for concise
  diagnostics.
- Hardened static typing across algorithms, Lyapunov solvers, problem classes,
  and shared utilities (including explicit backend protocol annotations and
  typed certificate/result internals).
- Tightened algorithm constructor/setter validation and shared helper
  utilities for dimensions and matrix checks.
- Added a default MOSEK Fusion parameter profile and normalized CVXPY solver
  parameter handling (including `COPT`/`MOSEK`-specific parameter normalization
  rules).
- Standardized iteration-dependent scalar naming to `c_K` across APIs, docs,
  and generated assets.
- Improved documentation structure and navigation (quick start, API layout,
  release notes organization, and examples indexing).
- Clarified theory documentation by centralizing shared solution notation and
  tightening theorem naming/cross-references across Lyapunov analysis pages.
- Expanded developer-internal documentation inventories to cover package-level
  exports and backend typing helpers.
- Switched docs builds from `html` to `dirhtml` across local commands,
  contributor docs, and release publication automation.
- Updated docs build behavior to use `.venv-docs` when available and isolate
  doctree caches by Sphinx version.
- Updated CI/release workflows to enforce `CITATION.cff` and `VERSION`
  consistency.
- Expanded lint/type-check policy coverage for core solver and Lyapunov modules.
- Refined API docstrings to link computational entry points with the new theory
  pages and matrix-form definitions.
- Expanded and reorganized test coverage across algorithms, convergence checks,
  backends, and problem-class validation.
- Added regression coverage for CVXPY inaccurate-status handling under
  `cvxpy_accept_inaccurate`.
- Updated package metadata and dependency model:
  - package version now sourced from `VERSION`.
  - core requirements now use `requirements.txt`.
  - optional extras now cover `mosek`, `sdpa`, `sdpa_multiprecision`, `copt`,
    and `test`.
- Updated release/contributing guidance and local docs-build/test instructions.
- Updated copyright year range in `LICENSE` to `2025-2026`.

### Removed

- Deprecated Lyapunov alias methods:
  - `IterationIndependent.verify_iteration_independent_Lyapunov`
  - `IterationDependent.verify_iteration_dependent_Lyapunov`
- `autolyap.algorithms.deterministic_proxskip` module and `ProxSkip` export.
- Stale `FullExtragradient` export from `autolyap.algorithms.__all__`.
- Tracked generated docs build artifacts and stray temporary files/workflows.

### Fixed

- Quick-start citation snippet and direct-run imports in docs scripts.
- OGM plot labels and iteration-dependent constant naming consistency.
- High-gamma Douglas-Rachford MOSEK tolerance in convergence checks.
- Local `make check` parity with CI pytest markers and dependency guidance.

### Security

- Added gitleaks secret scanning and upgraded CodeQL automation.
- Hardened CI handling of the MOSEK license secret for backend tests.

### Breaking changes

- Renamed interpolation index tokens:
  - `i<j` -> `r1<r2`
  - `i!=j` -> `r1!=r2`
  - `i` -> `r1`
  - `i!=star` -> `r1!=star`
- Renamed Lyapunov entry points:
  - `IterationIndependent.verify_iteration_independent_Lyapunov` ->
    `IterationIndependent.search_lyapunov`
  - `IterationDependent.verify_iteration_dependent_Lyapunov` ->
    `IterationDependent.search_lyapunov`
- Removed deprecated Lyapunov alias methods:
  - `IterationIndependent.verify_iteration_independent_Lyapunov`
  - `IterationDependent.verify_iteration_dependent_Lyapunov`
- Updated public return shapes for Lyapunov APIs:
  - `IterationIndependent.search_lyapunov` now returns
    `{"status", "solve_status", "rho", "certificate"}`
    (previously `bool`).
  - `IterationIndependent.LinearConvergence.bisection_search_rho` now returns
    `{"status", "solve_status", "rho", "certificate"}`
    (previously `float | None`).
  - `IterationDependent.search_lyapunov` now returns
    `{"status", "solve_status", "c_K", "certificate"}`
    (previously `(bool, c)`).
- Removed legacy `success` boolean signaling from Lyapunov outputs; callers
  should check `status == "feasible"` and inspect `solve_status`.
- Removed `autolyap.algorithms.deterministic_proxskip` / `ProxSkip`.

[Unreleased]: https://github.com/AutoLyap/AutoLyap/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/AutoLyap/AutoLyap/compare/v0.1.0...v0.2.0
