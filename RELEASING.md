# Releasing

## Versioning policy

AutoLyap uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `MAJOR`: incompatible API changes
- `MINOR`: backward-compatible functionality
- `PATCH`: backward-compatible bug fixes

## Release workflow

The `release` GitHub Actions workflow runs when a tag matching `v*` is pushed.
It runs the full test suite, validates that the tag matches the `VERSION` file,
builds source and wheel distributions, checks package metadata, creates a
GitHub Release with generated notes and attached artifacts, publishes to PyPI using a Trusted Publisher, and then publishes docs to
`AutoLyap/AutoLyap.github.io`.

The PyPI publish step is tied to the GitHub environment `pypi` and requires a
manual approval from configured reviewers.

Required GitHub Actions secrets:

- `AUTOLYAP_GH_PAGES_TOKEN` for docs publish to `AutoLyap/AutoLyap.github.io`.
- `MOSEK_LICENSE` for the `tests` workflow `pytest-mosek` job that runs on
  repository pushes and for release verification.

## Maintainer steps

1. Update `VERSION`.
2. Sync `CITATION.cff` version from `VERSION`:
   - `make sync-citation`
3. Update `CHANGELOG.md`.
4. Commit and push changes.
5. Merge to `main`.
6. Create and push a matching tag from the release commit on `main`:
   - `git tag v$(cat VERSION)`
   - `git push origin v$(cat VERSION)`
7. Approve the `pypi` environment deployment when prompted in GitHub Actions.
8. Verify docs were published to `https://autolyap.github.io/`.
