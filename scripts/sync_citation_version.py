#!/usr/bin/env python3
"""Sync the top-level version in CITATION.cff with VERSION.

Usage:
    python scripts/sync_citation_version.py
    python scripts/sync_citation_version.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _read_version(version_file: Path) -> str:
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Version file is empty: {version_file}")
    return version


def _extract_citation_version(citation_text: str) -> str:
    match = re.search(r"(?m)^version:\s*\"?([^\n\"]+)\"?\s*$", citation_text)
    if not match:
        raise ValueError("Could not find a top-level `version:` field in CITATION.cff")
    return match.group(1).strip()


def _replace_citation_version(citation_text: str, version: str) -> tuple[str, int]:
    updated_text, count = re.subn(
        r"(?m)^version:\s*\"?[^\n\"]+\"?\s*$",
        f'version: "{version}"',
        citation_text,
        count=1,
    )
    return updated_text, count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync CITATION.cff version with VERSION."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only; exit non-zero if CITATION.cff is out of sync.",
    )
    parser.add_argument(
        "--version-file",
        type=Path,
        default=Path("VERSION"),
        help="Path to VERSION file (default: VERSION).",
    )
    parser.add_argument(
        "--citation-file",
        type=Path,
        default=Path("CITATION.cff"),
        help="Path to CITATION.cff (default: CITATION.cff).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        version = _read_version(args.version_file)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"[sync-citation][error] {exc}", file=sys.stderr)
        return 1

    try:
        citation_text = args.citation_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"[sync-citation][error] Failed to read {args.citation_file}: {exc}", file=sys.stderr)
        return 1

    try:
        citation_version = _extract_citation_version(citation_text)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"[sync-citation][error] {exc}", file=sys.stderr)
        return 1

    if args.check:
        if citation_version != version:
            print(
                "[sync-citation][error] CITATION.cff version is out of sync: "
                f"CITATION.cff has {citation_version!r}, VERSION has {version!r}.",
                file=sys.stderr,
            )
            print(
                "Run: python scripts/sync_citation_version.py",
                file=sys.stderr,
            )
            return 1
        print(f"[sync-citation] OK: version is {version}.")
        return 0

    updated_text, replaced = _replace_citation_version(citation_text, version)
    if replaced != 1:
        print(
            "[sync-citation][error] Could not safely update exactly one `version:` field "
            f"in {args.citation_file}.",
            file=sys.stderr,
        )
        return 1

    if updated_text != citation_text:
        args.citation_file.write_text(updated_text, encoding="utf-8")
        print(
            f"[sync-citation] Updated {args.citation_file} version: "
            f"{citation_version!r} -> {version!r}"
        )
    else:
        print(f"[sync-citation] No change needed; version is already {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
