#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Audit generated documentation for stable, visual-neutral performance rules."""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from fontTools.ttLib import TTFont


SCRIPT_RE = re.compile(r"<script\b(?=[^>]*\bsrc=)[^>]*>", re.IGNORECASE)
IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ATTRIBUTE_TEMPLATE = r"\b{}\s*="
EXPECTED_MATH_TAG_PAGES = {
    "theory/iteration_dependent_analyses/index.html",
    "theory/iteration_independent_analyses/index.html",
    "theory/performance_estimation_via_sdps/index.html",
}
EXPECTED_MATH_HTML_PAGES = {
    "theory/iteration_independent_analyses/index.html",
}
FONT_BUDGETS = {
    "autolyap-lato-normal.woff2": 50_000,
    "autolyap-lato-bold.woff2": 50_000,
    "autolyap-lato-normal-italic.woff2": 50_000,
    "autolyap-lato-bold-italic.woff2": 50_000,
    "autolyap-fontawesome.woff2": 10_000,
}
REQUIRED_FONTAWESOME_CODEPOINTS = {
    0xF019,
    0xF02D,
    0xF057,
    0xF058,
    0xF06A,
    0xF08E,
    0xF0A8,
    0xF0A9,
    0xF0C1,
    0xF0C9,
    0xF0D7,
    0xF147,
    0xF196,
}


class VisibleTextParser(HTMLParser):
    """Collect visible characters while excluding scripts, styles, and icon PUA."""

    def __init__(self) -> None:
        super().__init__()
        self.codepoints: set[int] = set(range(0x20, 0x7F)) | {0x00A0}
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.hidden_depth:
            return
        self.codepoints.update(
            ord(character)
            for character in data
            if not 0xE000 <= ord(character) <= 0xF8FF
        )


def _has_attribute(tag: str, name: str) -> bool:
    return bool(re.search(ATTRIBUTE_TEMPLATE.format(re.escape(name)), tag, re.I))


def _relative_page(build_dir: Path, html_path: Path) -> str:
    return html_path.relative_to(build_dir).as_posix()


def _audit_html(build_dir: Path, html_paths: list[Path], errors: list[str]) -> None:
    math_tag_pages: set[str] = set()
    math_html_pages: set[str] = set()

    for html_path in html_paths:
        relative_page = _relative_page(build_dir, html_path)
        markup = html_path.read_text(encoding="utf-8")

        for script_tag in SCRIPT_RE.findall(markup):
            if not re.search(r"\b(?:defer|async)\b", script_tag, re.I):
                errors.append(f"{relative_page}: parser-blocking script: {script_tag}")

        if "jQuery(function" in markup:
            errors.append(
                f"{relative_page}: inline jQuery bootstrap races deferred scripts"
            )

        for image_tag in IMAGE_RE.findall(markup):
            missing = [
                name
                for name in ("width", "height", "loading", "decoding")
                if not _has_attribute(image_tag, name)
            ]
            if missing:
                errors.append(
                    f"{relative_page}: image missing {', '.join(missing)}: {image_tag}"
                )
            if "_images/" in image_tag:
                if 'loading="lazy"' not in image_tag:
                    errors.append(f"{relative_page}: plot image is not lazy-loaded")
                if 'fetchpriority="low"' not in image_tag:
                    errors.append(
                        f"{relative_page}: plot image lacks low fetch priority"
                    )
            if 'class="toggler"' in image_tag and (
                'loading="eager"' not in image_tag or 'decoding="sync"' not in image_tag
            ):
                errors.append(
                    f"{relative_page}: module-index toggle is not loaded eagerly"
                )

        has_code = "<pre" in markup
        has_pygments = "pygments.css" in markup
        has_copy_control = "copybutton.js" in markup
        if has_code and (not has_pygments or not has_copy_control):
            errors.append(
                f"{relative_page}: code page lacks Pygments CSS or copy controls"
            )
        if not has_code and (has_pygments or has_copy_control):
            errors.append(
                f"{relative_page}: code-only assets loaded without a code block"
            )

        if "math_tag_links.js" in markup:
            math_tag_pages.add(relative_page)
        if "[tex]/html" in markup:
            math_html_pages.add(relative_page)
        if "perf.js" in markup or "badge_links.js" in markup:
            errors.append(
                f"{relative_page}: obsolete runtime optimization script loaded"
            )
        if "favicon.ico" in markup:
            errors.append(f"{relative_page}: redundant legacy favicon loaded")

    if math_tag_pages != EXPECTED_MATH_TAG_PAGES:
        errors.append(
            "math_tag_links.js page set differs: "
            f"expected {sorted(EXPECTED_MATH_TAG_PAGES)}, got {sorted(math_tag_pages)}"
        )
    if math_html_pages != EXPECTED_MATH_HTML_PAGES:
        errors.append(
            "MathJax [tex]/html page set differs: "
            f"expected {sorted(EXPECTED_MATH_HTML_PAGES)}, got {sorted(math_html_pages)}"
        )


def _audit_fonts(build_dir: Path, html_paths: list[Path], errors: list[str]) -> None:
    font_dir = build_dir / "_static/fonts"
    for legacy_dir in (font_dir / "Lato", font_dir / "RobotoSlab"):
        if legacy_dir.exists():
            errors.append(f"unused theme font directory remains: {legacy_dir}")

    visible_parser = VisibleTextParser()
    for html_path in html_paths:
        visible_parser.feed(html_path.read_text(encoding="utf-8"))

    source_font_dir = build_dir / "_static/css/fonts"
    lato_sources = {
        "autolyap-lato-normal.woff2": "lato-normal.woff2",
        "autolyap-lato-bold.woff2": "lato-bold.woff2",
        "autolyap-lato-normal-italic.woff2": "lato-normal-italic.woff2",
        "autolyap-lato-bold-italic.woff2": "lato-bold-italic.woff2",
    }

    for subset_name, budget in FONT_BUDGETS.items():
        subset_path = font_dir / subset_name
        if not subset_path.is_file():
            errors.append(f"missing subset font: {subset_path}")
            continue
        if subset_path.stat().st_size > budget:
            errors.append(
                f"font exceeds {budget:,}-byte budget: {subset_name} "
                f"({subset_path.stat().st_size:,} bytes)"
            )
        if subset_path.read_bytes()[:4] != b"wOF2":
            errors.append(f"invalid WOFF2 signature: {subset_name}")

    for subset_name, source_name in lato_sources.items():
        subset_path = font_dir / subset_name
        source_path = source_font_dir / source_name
        if not subset_path.is_file() or not source_path.is_file():
            continue
        subset_cmap = set(TTFont(subset_path).getBestCmap())
        source_cmap = set(TTFont(source_path).getBestCmap())
        required = visible_parser.codepoints & source_cmap
        missing = required - subset_cmap
        if missing:
            errors.append(
                f"{subset_name}: missing visible glyphs "
                + ", ".join(f"U+{codepoint:04X}" for codepoint in sorted(missing))
            )

    awesome_path = font_dir / "autolyap-fontawesome.woff2"
    if awesome_path.is_file():
        awesome_cmap = set(TTFont(awesome_path).getBestCmap())
        missing = REQUIRED_FONTAWESOME_CODEPOINTS - awesome_cmap
        if missing:
            errors.append(
                "FontAwesome subset misses theme controls: "
                + ", ".join(f"U+{codepoint:04X}" for codepoint in sorted(missing))
            )


def _audit_artifacts(build_dir: Path, errors: list[str]) -> None:
    static_dir = build_dir / "_static"
    custom_css = (static_dir / "custom.css").read_text(encoding="utf-8")
    if 'font-family: "AutoLyap Lato Subset"' not in custom_css or (
        '"AutoLyap Lato Subset", "Lato"' not in custom_css
    ):
        errors.append("subset Lato lacks the full Lato fallback family")
    if 'img[src^="https://img.shields.io/"][width][height]' not in custom_css:
        errors.append("dynamic badges can be forced to a stale width")
    image_dir = build_dir / "_images"
    duplicate_svgs = {path.name for path in static_dir.glob("*.svg")} & {
        path.name for path in image_dir.glob("*.svg")
    }
    if duplicate_svgs:
        errors.append(f"plot SVGs deployed twice: {sorted(duplicate_svgs)}")
    for obsolete in (static_dir / "perf.js", static_dir / "badge_links.js"):
        if obsolete.exists():
            errors.append(f"obsolete static asset remains: {obsolete}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    html_paths = sorted(build_dir.rglob("*.html"))
    if not html_paths:
        print(f"No HTML pages found under {build_dir}", file=sys.stderr)
        return 2

    errors: list[str] = []
    _audit_html(build_dir, html_paths, errors)
    _audit_fonts(build_dir, html_paths, errors)
    _audit_artifacts(build_dir, errors)

    if errors:
        print("Documentation performance audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Documentation performance audit passed: {len(html_paths)} pages, "
        f"{sum(len(IMAGE_RE.findall(path.read_text('utf-8'))) for path in html_paths)} images."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
