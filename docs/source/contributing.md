# Contributing

Thank you for your interest in contributing to AutoLyap.
This guide explains how to set up a development environment, make changes, and
submit a high-quality pull request.

## Ways to contribute

You can contribute in several ways:

- Fix bugs or improve robustness.
- Add or improve algorithms, problem classes, and analyses.
- Improve documentation and examples.
- Add tests or improve existing test coverage.
- Report issues with clear reproduction steps.

## Development setup

Clone the repository and install in editable mode with test dependencies:

```bash
git clone https://github.com/AutoLyap/AutoLyap.git
cd AutoLyap
python -m pip install -e '.[test]'
```

AutoLyap requires Python `>=3.9`.

Install documentation dependencies (for docs work):

```bash
make -C docs deps
```

## Run tests

Run the full test suite:

```bash
python -m pytest
```

Run a specific test module:

```bash
python -m pytest tests/test_convergence_proximal_point.py -q
```

MOSEK-backed tests will skip automatically when MOSEK or a valid license is
not available. CVXPY backend smoke tests will skip when CVXPY or a supported
CVXPY SDP solver is unavailable.

## Build documentation

Build docs locally:

```bash
make -C docs html
```

Generated files are written to `docs/build/html/`.
Do not commit generated build artifacts from `docs/build/`.

## Contributing examples

Examples are especially welcome.

To add a new example:

1. Create a new page under `docs/source/examples/`.
2. Add it to the toctree in `docs/source/examples.md`.
3. Keep the example runnable and focused on one workflow.
4. Prefer built-in algorithm classes when available; use custom `Algorithm`
   subclasses when the example is specifically about defining a new method.

## Style and notation conventions

Please keep new contributions aligned with the current style and notation.

- Follow notation used in existing code and docs (symbols should keep the same
  meaning across files).
- Keep docstrings and docs consistent with existing section structure and
  wording style.
- When in doubt, follow nearby files.

## Recommended workflow

1. Sync `main`, then create a feature branch:

```bash
git switch main
git pull --rebase
git switch -c <feature-branch>
```

2. Implement your change in small, reviewable commits.
3. Run relevant tests locally.
4. Build docs if you changed any documentation.
5. Open a pull request with a clear summary and rationale.

## Pull request checklist

- [ ] Code changes are scoped and readable.
- [ ] Tests pass locally (`python -m pytest`).
- [ ] New behavior is covered by tests when applicable.
- [ ] Docs are updated when APIs/behavior changed.
- [ ] Docs build locally (`make -C docs html`) if docs were changed.
- [ ] No generated files from `docs/build/` are included in the commit.
- [ ] PR description states what changed, why, and which test and docs build commands were run.

## Reporting issues

For bug reports and feature requests, open an issue at:

<https://github.com/AutoLyap/AutoLyap/issues>
