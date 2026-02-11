# Releasing

## Versioning policy

AutoLyap uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `MAJOR`: incompatible API changes
- `MINOR`: backward-compatible functionality
- `PATCH`: backward-compatible bug fixes

## Release workflow

The `release` GitHub Actions workflow runs when a tag matching `v*` is pushed.
It validates that the tag matches the `VERSION` file, builds source and wheel
distributions, checks package metadata, and creates a GitHub Release with
generated notes and attached artifacts.

## Maintainer steps

1. Update `VERSION`.
2. Update `CHANGELOG.md`.
3. Commit and push changes.
4. Create and push a matching tag:
   - `git tag v$(cat VERSION)`
   - `git push origin v$(cat VERSION)`
