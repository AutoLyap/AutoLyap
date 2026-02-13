# Test Suite Layout

Tests are grouped by domain:

- `tests/algorithm/`: algorithm construction, matrix generation, setters, and caching.
- `tests/backend/`: backend behavior (CVXPY/MOSEK), backend equivalence, and backend builders.
- `tests/convergence/`: convergence checks against theory (including CVXPY-specific convergence tests).
- `tests/problemclass/`: function/operator conditions, interpolation indices, and inclusion-problem validation/updates.
- `tests/lyapunov/`: iteration-dependent/independent Lyapunov parameter checks.
- `tests/solver/`: solver option normalization and policy checks.

Docs-derived tests are grouped by technical domain instead of a separate docs folder.
Use a `_docs_example` suffix in the filename (for example,
`test_convergence_cvxpy_proximal_gradient_docs_example.py`) to keep origin traceable.

Shared support modules:

- `tests/conftest.py`: common algorithm fixtures used across test modules.
- `tests/shared/mosek_utils.py`: MOSEK license checks for MOSEK-dependent tests.
- `tests/shared/cvxpy_test_utils.py`: CVXPY/MOSEK option-builder helpers.
- `tests/shared/cvxpy_fixtures.py`: reusable CVXPY/MOSEK fixtures for backend/convergence tests.
- `tests/convergence/convergence_douglas_rachford_utils.py`: Douglas-Rachford theory formulas and shared bisection wrapper.

# Public vs Private API Tests

Tests are also separated by API surface:

- `public_api`: validates user-facing behavior and stable contracts.
- `private_api`: validates internal implementation details (private helpers, internal builders, and internals-only invariants).

Markers are assigned automatically in `tests/conftest.py` using folder/file routing, so tests usually do not need explicit `@pytest.mark.*` decorations.

Useful commands:

- Run only public API tests: `pytest -m public_api`
- Run only private API tests: `pytest -m private_api`
- Exclude private API tests: `pytest -m "not private_api"`
