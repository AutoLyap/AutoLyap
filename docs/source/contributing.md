# Contributing

Thank you for your interest in contributing to AutoLyap. This guide is for contributors working from a fork and submitting pull requests.

## Source code 

The project is hosted on GitHub at [https://github.com/AutoLyap/AutoLyap](https://github.com/AutoLyap/AutoLyap).

## Ways to contribute

- Fix bugs or improve robustness.
- Add or improve algorithms, problem classes, analyses, and examples.
- Improve documentation.
- Add tests or improve existing test coverage.
- Report issues with clear reproduction steps.

## Before you start

- For substantial changes, open an issue first so scope and direction are clear.
- For small fixes (typos, minor docs, straightforward bug fix), you can open a PR directly.
- If an issue exists, reference it in your PR description.

Issue tracker:
<https://github.com/AutoLyap/AutoLyap/issues>

## Fork-first workflow

1. Fork `AutoLyap/AutoLyap` on GitHub.
2. Clone your fork and add the upstream remote:

```bash
git clone https://github.com/<your-github-username>/AutoLyap.git
cd AutoLyap
git remote add upstream https://github.com/AutoLyap/AutoLyap.git
git fetch upstream
```

3. Create a feature branch from `upstream/main`:

```bash
git switch -c <feature-branch> upstream/main
```

4. Install in editable mode with test dependencies:

```bash
python -m pip install -e '.[test]'
```

AutoLyap requires Python `>=3.9`.

## Keep your branch current

Rebase your branch onto `upstream/main` before opening or updating a PR:

```bash
git fetch upstream
git rebase upstream/main
```

## Local checks before opening a PR

Run the one-command local CI helper:

```bash
make check
```

If needed, install missing lint/typecheck dependencies:

```bash
python -m pip install ruff mypy
```

The CI test matrix runs on Python `3.9`, `3.10`, `3.11`, `3.12`, and `3.13`.
At minimum, run the checks above locally on one supported Python version.

If your change affects MOSEK, run:

```bash
make check-mosek
```

## Notes for docs

- Keep docstrings, docs, and mathematical notation consistent with existing structure and style.
- To build the Sphinx documentation locally:

```bash
make -C docs deps
make -C docs html
```

The generated site is written to `docs/build/html/` (open `docs/build/html/index.html`).

## Open a pull request

1. Push your branch to your fork:

```bash
git push -u origin <feature-branch>
```

2. Open a PR from your fork branch to `AutoLyap/AutoLyap:main`.

## Pull request checklist

- [ ] Code changes are scoped and readable.
- [ ] CI-equivalent local checks pass.
- [ ] New behavior is covered by tests when applicable.
- [ ] Docs are updated when API or behavior changed.
- [ ] Docs build locally (`make -C docs html`) if docs were changed.
- [ ] No generated files from `docs/build/` are committed.
- [ ] PR description states what changed and why.
