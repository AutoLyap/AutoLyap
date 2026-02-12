# Releasing

## Versioning policy

AutoLyap uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `MAJOR`: incompatible API changes
- `MINOR`: backward-compatible functionality
- `PATCH`: backward-compatible bug fixes

## Release workflow

The `release` GitHub Actions workflow runs when a tag matching `v*` is pushed.
It runs the core test suite, validates that the tag matches the `VERSION` file,
builds source and wheel distributions, checks package metadata, creates a
GitHub Release with generated notes and attached artifacts, and then publishes
to PyPI using a Trusted Publisher.

The PyPI publish step is tied to the GitHub environment `pypi` and requires a
manual approval from configured reviewers.

## Maintainer steps

1. Update `VERSION`.
2. Update `CHANGELOG.md`.
3. Commit and push changes.
4. Merge to `main`.
5. Create and push a matching tag from the release commit on `main`:
   - `git tag v$(cat VERSION)`
   - `git push origin v$(cat VERSION)`
6. Approve the `pypi` environment deployment when prompted in GitHub Actions.
