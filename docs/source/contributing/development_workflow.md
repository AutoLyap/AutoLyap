# Development workflow

## Keep your branch current

Rebase your branch onto `upstream/main` before opening or updating a PR:

```bash
git fetch upstream
git rebase upstream/main
```

## Run local checks before opening a PR

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

If your change may impact MOSEK-backed functionality, run:

```bash
make check-mosek
```

## Documentation changes

- Keep docstrings, docs, and mathematical notation consistent with existing structure and style.
- To build the documentation locally:

```bash
make -C docs deps
make -C docs html
```

The generated site is written to `docs/build/html/` (open `docs/build/html/index.html`).
