# Getting started

## Repository and issue tracker

- The project is hosted on GitHub at [https://github.com/AutoLyap/AutoLyap](https://github.com/AutoLyap/AutoLyap).
- Issue tracker:
<https://github.com/AutoLyap/AutoLyap/issues>

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

## Set up your fork workspace

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
