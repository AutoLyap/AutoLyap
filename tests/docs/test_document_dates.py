# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

# Check rendered-content dates using real temporary Git repositories.

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.docs_dates import document_source_dates


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def commit(repo, path, content, timestamp):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "--", path], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Docs Dates",
            "-c",
            "user.email=docs-dates@example.invalid",
            "commit",
            "-q",
            "-m",
            "Update content",
        ],
        cwd=repo,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": timestamp,
            "GIT_COMMITTER_DATE": timestamp,
        },
        check=True,
    )


def make_app(repo, *, dependencies=(), absolute_source=True):
    srcdir = repo / "docs/source"
    return SimpleNamespace(
        env=SimpleNamespace(
            srcdir=srcdir,
            doc2path=lambda docname: (
                srcdir / "index.md" if absolute_source else "index.md"
            ),
            dependencies={"index": set(dependencies)},
        ),
    )


@pytest.mark.parametrize(
    "dependency", ["README.md", "autolyap/example.py", "docs/source/references.bib"]
)
@pytest.mark.parametrize("absolute", [True, False])
def test_newer_content_dependency_updates_modification_date(
    git_repo, dependency, absolute
):
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-02T03:04:05+00:00")
    commit(git_repo, dependency, "Original content", "2021-02-03T04:05:06+00:00")
    commit(git_repo, dependency, "Updated content", "2023-06-07T08:09:10+02:00")
    path = git_repo / dependency
    if not absolute:
        path = Path(os.path.relpath(path, git_repo / "docs/source"))
    app = make_app(git_repo, dependencies=[path], absolute_source=absolute)

    assert document_source_dates(app, "index", repo_root=git_repo) == (
        "2020-01-02T03:04:05Z",
        "2023-06-07T06:09:10Z",
    )


def test_wrapper_history_preserves_publication_and_newest_modification(git_repo):
    commit(git_repo, "README.md", "Earlier dependency", "2019-01-01T00:00:00+00:00")
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-02T03:04:05+00:00")
    commit(
        git_repo, "docs/source/index.md", "Updated page", "2024-01-01T00:00:00+00:00"
    )
    app = make_app(git_repo, dependencies=["../../README.md"])

    assert document_source_dates(app, "index", repo_root=git_repo) == (
        "2020-01-02T03:04:05Z",
        "2024-01-01T00:00:00Z",
    )


def test_version_change_updates_rendered_header_modification(git_repo):
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-02T03:04:05+00:00")
    commit(git_repo, "VERSION", "1.0", "2024-01-01T00:00:00+00:00")

    assert document_source_dates(make_app(git_repo), "index", repo_root=git_repo) == (
        "2020-01-02T03:04:05Z",
        "2024-01-01T00:00:00Z",
    )


def test_external_missing_and_untracked_dependencies_do_not_supply_dates(git_repo):
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-02T03:04:05+00:00")
    commit(
        git_repo,
        "docs/source/missing.md",
        "Deleted dependency",
        "2024-01-01T00:00:00+00:00",
    )
    (git_repo / "docs/source/missing.md").unlink()
    (git_repo / "docs/source/untracked.md").write_text("Untracked", encoding="utf-8")
    external = git_repo.parent / "external"
    subprocess.run(["git", "init", "-q", str(external)], check=True)
    commit(external, "outside.md", "External", "2025-01-01T00:00:00+00:00")
    (git_repo / "outside-link.md").symlink_to(external / "outside.md")
    app = make_app(
        git_repo,
        dependencies=[
            "missing.md",
            "untracked.md",
            external / "outside.md",
            git_repo / "outside-link.md",
            git_repo / "docs/source",
        ],
    )

    assert document_source_dates(app, "index", repo_root=git_repo) == (
        "2020-01-02T03:04:05Z",
        "2020-01-02T03:04:05Z",
    )


@pytest.mark.parametrize("with_git", [True, False])
def test_unknown_history_omits_checkout_dates(tmp_path, with_git):
    repo = tmp_path / "empty"
    source = repo / "docs/source/index.md"
    source.parent.mkdir(parents=True)
    source.write_text("Page without history", encoding="utf-8")
    if with_git:
        subprocess.run(["git", "init", "-q", str(repo)], check=True)

    assert document_source_dates(make_app(repo), "index", repo_root=repo) == ("", "")


def test_known_dependency_does_not_invent_wrapper_publication(git_repo):
    commit(git_repo, "README.md", "Included content", "2024-01-01T00:00:00+00:00")
    source = git_repo / "docs/source/index.md"
    source.parent.mkdir(parents=True)
    source.write_text("Untracked wrapper", encoding="utf-8")
    app = make_app(git_repo, dependencies=["../../README.md"])

    assert document_source_dates(app, "index", repo_root=git_repo) == (
        "",
        "2024-01-01T00:00:00Z",
    )


def test_modification_is_maximum_timestamp_even_with_git_clock_skew(git_repo):
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-02T03:04:05+00:00")
    commit(git_repo, "README.md", "Dependency", "2024-01-01T00:00:00+00:00")
    commit(
        git_repo,
        "README.md",
        "Dependency with earlier clock",
        "2021-01-01T00:00:00+00:00",
    )
    app = make_app(git_repo, dependencies=["../../README.md"])

    assert document_source_dates(app, "index", repo_root=git_repo) == (
        "2020-01-02T03:04:05Z",
        "2024-01-01T00:00:00Z",
    )


def test_cached_history_is_scoped_to_repository(git_repo):
    commit(git_repo, "docs/source/index.md", "First page", "2020-01-01T00:00:00+00:00")
    other = git_repo.parent / "other"
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    commit(other, "docs/source/index.md", "Second page", "2024-01-01T00:00:00+00:00")

    assert document_source_dates(make_app(git_repo), "index", repo_root=git_repo) == (
        "2020-01-01T00:00:00Z",
        "2020-01-01T00:00:00Z",
    )
    assert document_source_dates(make_app(other), "index", repo_root=other) == (
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    )
    (git_repo / ".git").rename(git_repo / "cached-history")
    assert document_source_dates(make_app(git_repo), "index", repo_root=git_repo) == (
        "2020-01-01T00:00:00Z",
        "2020-01-01T00:00:00Z",
    )


def test_unavailable_git_omits_dates(git_repo, monkeypatch):
    commit(git_repo, "docs/source/index.md", "Page", "2020-01-01T00:00:00+00:00")
    monkeypatch.setenv("PATH", "")

    assert document_source_dates(make_app(git_repo), "index", repo_root=git_repo) == (
        "",
        "",
    )
