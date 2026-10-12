# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Derive documentation dates from repository-owned content dependencies."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


def _as_utc_iso(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@lru_cache(maxsize=None)
def _git_source_dates(repo_root: str, relative_path: str) -> tuple[str, str]:
    """Cache the first and maximum Git timestamps by repository and file."""
    try:
        result = subprocess.run(
            [
                "git",
                "--literal-pathspecs",
                "log",
                "--follow",
                "--format=%cI",
                "--",
                relative_path,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "", ""
    if result.returncode:
        return "", ""
    history = [
        normalized
        for line in result.stdout.splitlines()
        if (normalized := _as_utc_iso(line))
    ]
    return (history[-1], max(history)) if history else ("", "")


def _repository_file(path, *, srcdir: Path, repo_root: Path) -> Path | None:
    try:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = srcdir / resolved
        resolved = resolved.resolve()
        resolved.relative_to(repo_root)
        return resolved if resolved.is_file() else None
    except (OSError, TypeError, ValueError):
        return None


def document_source_dates(app, docname: str, *, repo_root) -> tuple[str, str]:
    """Return wrapper publication and latest known rendered-content Git dates.

    Relative source and dependency paths are interpreted against ``env.srcdir``.
    Unknown Git history contributes no date; checkout times are never used.
    ``VERSION`` contributes to modification because it appears in the page header.
    """
    repo_root = Path(repo_root).resolve()
    env = app.env
    srcdir = Path(env.srcdir).resolve()
    try:
        source = _repository_file(
            env.doc2path(docname), srcdir=srcdir, repo_root=repo_root
        )
    except Exception:
        return "", ""
    published = ""
    paths = set()
    if source is not None:
        paths.add(source)
        published, _ = _git_source_dates(
            str(repo_root), source.relative_to(repo_root).as_posix()
        )
    for dependency in getattr(env, "dependencies", {}).get(docname, ()):
        path = _repository_file(dependency, srcdir=srcdir, repo_root=repo_root)
        if path is not None:
            paths.add(path)
    version = _repository_file(
        repo_root / "VERSION", srcdir=srcdir, repo_root=repo_root
    )
    if version is not None:
        paths.add(version)
    modified = max(
        (
            _git_source_dates(str(repo_root), path.relative_to(repo_root).as_posix())[1]
            for path in paths
        ),
        default="",
    )
    return published, modified
