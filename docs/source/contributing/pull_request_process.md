# Pull request process

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
- [ ] Docs build locally (`make -C docs dirhtml`) if docs were changed.
- [ ] No generated files from `docs/build/` are committed.
- [ ] PR description states what changed and why.
