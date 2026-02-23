# Developer Commands (Internal)

This file documents developer-facing commands implemented in this repository.
It is intentionally separate from user documentation.

Before running the commands below, install the project in editable mode:

```bash
python -m pip install -e '.[test]'
```

If you need MOSEK-backed tests/examples, also install:

```bash
python -m pip install -e '.[mosek]'
```

If you need SDPA-backed convergence checks, install:

```bash
python -m pip install -e '.[sdpa]'
python -m pip install -e '.[sdpa_multiprecision]'  # for multiprecision checks
```

For documentation builds, install docs dependencies:

```bash
make -C docs deps
```

## Project-level Make targets

### `make check`

- Runs local CI checks via `bash scripts/check_local_ci.sh`.
- Sequence:
  1. `python scripts/sync_citation_version.py --check`.
  2. `ruff` fatal checks (`E9`, `F63`, `F7`, `F82`).
  3. Core lint policy checks on key modules via `ruff check`.
  4. `mypy` on selected modules.
  5. `pytest -m "not mosek and not sdpa and not sdpa_multiprecision"`.
  6. Strict `pytest -m "mosek"` validation (fails if MOSEK is unavailable or MOSEK tests are skipped).
- Fails fast on missing `ruff`/`mypy`.
- Side effects: creates temporary files for pytest XML reports when needed.

### `make check-mosek`

- Runs strict MOSEK validation via `bash scripts/check_local_ci.sh --mosek-only`.
- Requires a working MOSEK installation and valid license.
- Enforces:
  - MOSEK-marked tests run.
  - Zero skipped tests in the MOSEK suite.
- Intended for changes affecting MOSEK-backed behavior.

### `make check-sdpa`

- Runs SDPA-marked convergence tests via:
  `python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa and not sdpa_multiprecision"`.
- Intended for optional CVXPY+SDPA regular-precision validation.
- Not part of the default `make check` suite.

### `make check-sdpa-multiprecision`

- Runs SDPA multiprecision-marked convergence tests via:
  `python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa_multiprecision"`.
- Intended for optional CVXPY+SDPA multiprecision validation.
- Not part of the default `make check` suite.

### `make docs`

- Builds documentation with `make -C docs dirhtml`.
- Requires docs dependencies (install with `make -C docs deps`).
- Output: `docs/build/dirhtml/`.
- Inspect: `docs/build/dirhtml/index.html`.

### `make sync-citation`

- Syncs `CITATION.cff` top-level `version:` field with `VERSION`.
- Uses: `python scripts/sync_citation_version.py`.
- Run this after changing `VERSION`.

### `make check-citation`

- Verifies `CITATION.cff` is in sync with `VERSION` without modifying files.
- Uses: `python scripts/sync_citation_version.py --check`.
- Exits non-zero if versions differ.

## Script entry points

### `bash scripts/check_local_ci.sh`

- Same checks as `make check`.
- Options:
  - `--mosek-only`: strict MOSEK-only mode (same as `make check-mosek`).
  - `-h` / `--help`: usage.

### `python scripts/test_verbosity_outputs.py`

- Runs sanity checks for verbosity behavior across:
  - Iteration-independent `search_lyapunov`.
  - Iteration-independent `bisection_search_rho`.
  - Iteration-dependent `search_lyapunov`.
- Tests verbosity levels `0`, `1`, and `2`.
- Returns non-zero if any run fails or returns non-feasible status.
- Useful for validating logging/verbosity changes.

Options:

- `--backend {cvxpy,mosek_fusion}` (default: `cvxpy`)
- `--cvxpy-solver <SOLVER_NAME>`
- `--max-iter <INT>` (forwarded to solver-specific CVXPY params when supported)

### `python scripts/sync_citation_version.py`

- Syncs `CITATION.cff` with `VERSION`.
- Options:
  - `--check`: validate only, no file changes.
  - `--version-file <PATH>`: alternate `VERSION` file.
  - `--citation-file <PATH>`: alternate `CITATION.cff`.

## Docs plot/data generation scripts

The docs include script-driven generation of CSV data tables and SVG plots under
`docs/source/data/` and `docs/source/_static/`.

Common behavior for all `generate_*_assets.py` scripts:

- Backend is `mosek_fusion` (required for docs assets in this repo).
- `--output-dir <DIR>` defaults to `docs/source`.
- `--reuse-data` skips expensive solves and regenerates SVG files from existing CSVs.

Folder structure:

```text
docs/source/examples/scripts/
  shared/
    plotting_utils.py
  gradient_method/
  proximal_gradient/
  proximal_point/
  chambolle_pock/
  heavy_ball/
  nesterov_momentum/
  optimized_gradient/
  nesterov_fast_gradient/
  douglas_rachford/
```

Run from repository root:

```bash
python docs/source/examples/scripts/gradient_method/generate_gradient_method_assets.py
python docs/source/examples/scripts/proximal_gradient/generate_proximal_gradient_assets.py
python docs/source/examples/scripts/proximal_point/generate_proximal_point_assets.py
python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_assets.py --parallel-layers
python docs/source/examples/scripts/heavy_ball/generate_heavy_ball_smooth_convex_assets.py
python docs/source/examples/scripts/heavy_ball/generate_heavy_ball_gradient_dominated_smooth_assets.py
python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_smooth_convex_assets.py
python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_gradient_dominated_smooth_assets.py
python docs/source/examples/scripts/optimized_gradient/generate_optimized_gradient_assets.py
python docs/source/examples/scripts/nesterov_fast_gradient/generate_nesterov_fast_gradient_assets.py
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_lipschitz_assets.py
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_cocoercive_assets.py
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_cocoercive_plus_strongly_monotone_assets.py
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_lipschitz_plus_strongly_monotone_assets.py
```

Script-specific notable options:

- `generate_gradient_method_assets.py`: `--mu`, `--L`
- `generate_proximal_gradient_assets.py`: `--mu`, `--L`
- `generate_proximal_point_assets.py`: `--mu`
- `generate_chambolle_pock_assets.py`: `--parallel-layers`, `--max-workers`
- `generate_heavy_ball_smooth_convex_assets.py`: `--L`
- `generate_heavy_ball_gradient_dominated_smooth_assets.py`: `--mu-gd`, `--L`
- `generate_nesterov_momentum_smooth_convex_assets.py`: `--L`
- `generate_nesterov_momentum_gradient_dominated_smooth_assets.py`: `--mu-gd`, `--L`
- `generate_optimized_gradient_assets.py`: `--L`, `--k-min`, `--k-max`
- `generate_nesterov_fast_gradient_assets.py`: `--L`, `--k-min`, `--k-max`
- `generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_lipschitz_assets.py`: `--mu`, `--L`, `--lambda-value`
- `generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_cocoercive_assets.py`: `--mu`, `--beta`, `--lambda-value`
- `generate_douglas_rachford_cocoercive_plus_strongly_monotone_assets.py`: `--mu`, `--beta`, `--gamma`
- `generate_douglas_rachford_maximally_monotone_lipschitz_plus_strongly_monotone_assets.py`: `--mu`, `--L`, `--gamma`

Quick SVG-only refresh (existing CSV data required):

```bash
python docs/source/examples/scripts/gradient_method/generate_gradient_method_assets.py --reuse-data
python docs/source/examples/scripts/proximal_gradient/generate_proximal_gradient_assets.py --reuse-data
python docs/source/examples/scripts/proximal_point/generate_proximal_point_assets.py --reuse-data
python docs/source/examples/scripts/chambolle_pock/generate_chambolle_pock_assets.py --reuse-data
python docs/source/examples/scripts/heavy_ball/generate_heavy_ball_smooth_convex_assets.py --reuse-data
python docs/source/examples/scripts/heavy_ball/generate_heavy_ball_gradient_dominated_smooth_assets.py --reuse-data
python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_smooth_convex_assets.py --reuse-data
python docs/source/examples/scripts/nesterov_momentum/generate_nesterov_momentum_gradient_dominated_smooth_assets.py --reuse-data
python docs/source/examples/scripts/optimized_gradient/generate_optimized_gradient_assets.py --reuse-data
python docs/source/examples/scripts/nesterov_fast_gradient/generate_nesterov_fast_gradient_assets.py --reuse-data
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_lipschitz_assets.py --reuse-data
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_plus_strongly_monotone_cocoercive_assets.py --reuse-data
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_cocoercive_plus_strongly_monotone_assets.py --reuse-data
python docs/source/examples/scripts/douglas_rachford/generate_douglas_rachford_maximally_monotone_lipschitz_plus_strongly_monotone_assets.py --reuse-data
```

## Docs Make targets

From the repository root, use `make -C docs <target>`.

### `make -C docs deps`

- Installs docs dependencies from `docs/requirements.txt`.

### `make -C docs check-deps`

- Verifies required docs packages are importable.

### `make -C docs dirhtml`

- Builds Sphinx directory-style HTML docs into `docs/build/dirhtml/`.
- Inspect: `docs/build/dirhtml/index.html`.

### `make -C docs clean`

- Removes docs build artifacts (`docs/build/`).
