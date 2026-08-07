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
from urllib.parse import urlparse

from fontTools.ttLib import TTFont


SCRIPT_RE = re.compile(r"<script\b(?=[^>]*\bsrc=)[^>]*>", re.IGNORECASE)
IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
CONTAINER_RE = re.compile(r"<(?:div|span)\b[^>]*>", re.IGNORECASE)
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
    "autolyap-lato-normal.woff2": 16_000,
    "autolyap-lato-bold.woff2": 16_000,
    "autolyap-lato-normal-italic.woff2": 17_000,
    "autolyap-lato-bold-italic.woff2": 17_000,
    "autolyap-lato-greek-normal.woff2": 12_000,
    "autolyap-lato-greek-bold.woff2": 12_000,
    "autolyap-fontawesome.woff2": 2_500,
}
DOMAIN_GREEK_CODEPOINTS = set(map(ord, "ΓΔΘΛΞΟΠΣΦΨΩαβγδεζηθικλμνξοπρστυφχψω"))
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


def _attribute_value(tag: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*([\"'])(.*?)\1",
        tag,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(2) if match else ""


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

        first_script_index = markup.find("<script")
        for preload_name in (
            "autolyap-lato-normal.woff2",
            "autolyap-lato-bold.woff2",
            "autolyap-fontawesome.woff2",
        ):
            preload_index = markup.find(preload_name)
            if preload_index < 0 or preload_index > first_script_index:
                errors.append(
                    f"{relative_page}: critical font preload is discovered late: "
                    f"{preload_name}"
                )

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
            image_hostname = urlparse(_attribute_value(image_tag, "src")).hostname
            if image_hostname == "img.shields.io" and (
                'fetchpriority="low"' not in image_tag
            ):
                errors.append(
                    f"{relative_page}: remote badge lacks low fetch priority"
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
        if "_sphinx_javascript_frameworks_compat.js" in markup:
            errors.append(f"{relative_page}: unused Sphinx jQuery shim loaded")
        if re.search(r'<span\s+class=["\']pre["\']>', markup, flags=re.IGNORECASE):
            errors.append(f"{relative_page}: neutral Sphinx literal wrappers remain")
        if re.search(
            r'<span\s+class=["\'](?:n|p)["\']>', markup, flags=re.IGNORECASE
        ):
            errors.append(f"{relative_page}: neutral Pygments token wrappers remain")
        if re.search(
            r'<span\s+class=["\']w["\']>\s*</span>', markup, flags=re.IGNORECASE
        ):
            errors.append(f"{relative_page}: styled whitespace wrappers remain")
        if re.search(r"<span\s*></span>", markup, flags=re.IGNORECASE):
            errors.append(f"{relative_page}: empty generated spans remain")
        if "favicon.ico" in markup:
            errors.append(f"{relative_page}: redundant legacy favicon loaded")
        if re.search(r"<math\b", markup, flags=re.IGNORECASE):
            errors.append(f"{relative_page}: raw MathML requires the removed input component")
        if 'role="doc-biblioentry"' in markup:
            errors.append(f"{relative_page}: deprecated bibliography role remains")
        if re.search(
            r"<dl>\s*<dd><em class=[\"']sig-param[\"']",
            markup,
            flags=re.IGNORECASE,
        ):
            errors.append(f"{relative_page}: API parameter list lacks a semantic term")

        math_containers = []
        for tag in CONTAINER_RE.findall(markup):
            class_match = re.search(
                r'\bclass\s*=\s*(["\'])(.*?)\1',
                tag,
                flags=re.IGNORECASE | re.DOTALL,
            )
            classes = class_match.group(2).split() if class_match else []
            if "math" in classes:
                math_containers.append(classes)
        if math_containers:
            if "tex-mml-chtml.js" in markup or "tex-chtml.js" not in markup:
                errors.append(f"{relative_page}: MathJax includes unused input components")
            mathjax_script = re.search(
                r'<script\b(?=[^>]*\btex-chtml\.js)(?=[^>]*\bcrossorigin=["\']anonymous["\'])[^>]*>',
                markup,
                flags=re.IGNORECASE,
            )
            if mathjax_script is None:
                errors.append(f"{relative_page}: MathJax cannot reuse the CDN preconnect")
            preconnect_index = markup.find(
                '<link rel="preconnect" href="https://cdn.jsdelivr.net"'
            )
            mathjax_index = markup.find("tex-chtml.js")
            if preconnect_index < 0 or preconnect_index > mathjax_index:
                errors.append(f"{relative_page}: MathJax CDN preconnect is discovered late")
            zero_font_preload = "MathJax_Zero.woff\" as=\"font\"" in markup
            expects_zero_font_preload = (
                relative_page
                == "theory/iteration_independent_analyses/index.html"
            )
            if zero_font_preload != expects_zero_font_preload:
                errors.append(
                    f"{relative_page}: desktop MathJax zero-font preload differs"
                )
            if zero_font_preload and 'media="(min-width: 769px)"' not in markup:
                errors.append(
                    f"{relative_page}: MathJax zero-font preload is not desktop-only"
                )
            has_math_loading_guard = (
                'classList.add("autolyap-math-loading")' in markup
                and 'classList.remove("autolyap-math-loading")' in markup
                and ",8000);" in markup
            )
            if has_math_loading_guard != expects_zero_font_preload:
                errors.append(
                    f"{relative_page}: bounded MathJax loading guard differs"
                )
            expected_eager = min(4, len(math_containers))
            eager_prefix = sum(
                "math-initial" in classes
                for classes in math_containers[:expected_eager]
            )
            total_eager = sum(
                "math-initial" in classes for classes in math_containers
            )
            if eager_prefix != expected_eager or total_eager != expected_eager:
                errors.append(
                    f"{relative_page}: expected exactly {expected_eager} initial math "
                    f"containers, got prefix={eager_prefix}, total={total_eager}"
                )
            uses_lazy = '"ui/lazy"' in markup
            if len(math_containers) > expected_eager and (
                not uses_lazy or '"lazyAlwaysTypeset"' not in markup
            ):
                errors.append(
                    f"{relative_page}: MathJax lazy typesetting is not configured"
                )
            if len(math_containers) == expected_eager and uses_lazy:
                errors.append(
                    f"{relative_page}: MathJax lazy component is redundant"
                )

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

    for greek_name in (
        "autolyap-lato-greek-normal.woff2",
        "autolyap-lato-greek-bold.woff2",
    ):
        greek_path = font_dir / greek_name
        if not greek_path.is_file():
            continue
        greek_cmap = set(TTFont(greek_path).getBestCmap())
        missing = DOMAIN_GREEK_CODEPOINTS - greek_cmap
        if missing:
            errors.append(
                f"{greek_name}: missing domain Greek glyphs "
                + ", ".join(f"U+{codepoint:04X}" for codepoint in sorted(missing))
            )


def _audit_artifacts(build_dir: Path, errors: list[str]) -> None:
    static_dir = build_dir / "_static"
    custom_css = (static_dir / "custom.css").read_text(encoding="utf-8")
    custom_css_path = static_dir / "custom.css"
    if custom_css_path.stat().st_size > 21_000:
        errors.append(
            "minified custom CSS exceeds 21,000-byte budget: "
            f"{custom_css_path.stat().st_size:,} bytes"
        )
    pygments_css_path = static_dir / "pygments.css"
    if pygments_css_path.is_file() and pygments_css_path.stat().st_size > 1_300:
        errors.append(
            "pruned Pygments CSS exceeds 1,300-byte budget: "
            f"{pygments_css_path.stat().st_size:,} bytes"
        )
    theme_css_path = static_dir / "css" / "theme.css"
    theme_css = theme_css_path.read_text(encoding="utf-8")
    if theme_css_path.stat().st_size > 54_000:
        errors.append(
            "minified RTD theme exceeds 54,000-byte budget: "
            f"{theme_css_path.stat().st_size:,} bytes"
        )
    for selector in (
        ".current",
        ".fa-bars",
        ".rst-content",
        ".search",
        ".shift",
        ".shift-up",
        ".toctree-expand",
        ".wy-nav-top",
        ".wy-menu-vertical",
        ".wy-nav-content",
        ".wy-nav-side",
        ".wy-table-responsive",
    ):
        if selector not in theme_css:
            errors.append(f"pruned RTD theme lost critical selector: {selector}")
    if re.search(
        r"@font-face\{font-family:(?:FontAwesome|Roboto Slab)",
        theme_css,
        flags=re.IGNORECASE,
    ):
        errors.append("theme CSS retains superseded font faces")
    if not re.search(
        r'font-family:\s*"AutoLyap Lato Subset"', custom_css
    ) or not re.search(r'"AutoLyap Lato Subset"\s*,\s*"Lato"', custom_css):
        errors.append("subset Lato lacks the full Lato fallback family")
    if "autolyap-lato-greek-normal.woff2" not in custom_css or (
        "autolyap-lato-greek-bold.woff2" not in custom_css
    ):
        errors.append("runtime Greek search fonts are not declared")
    if ".autolyap-sr-only" not in custom_css:
        errors.append("wrapped API signatures lack screen-reader-only terms")
    compact_custom_css = re.sub(r"\s+", "", custom_css)
    for selector in (
        "html.autolyap-math-loading:has(#iteration-independent-analyses>div.math.math-initial",
        "html.autolyap-math-loading#iteration-independent-analyses>div.math.math-initial:not([id])",
        "html.autolyap-math-loading#equation-eq-constant-abcd.math-initial",
    ):
        if selector not in compact_custom_css:
            errors.append(f"initial theory math lacks geometry guard: {selector}")
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
    legacy_jquery_shim = static_dir / "_sphinx_javascript_frameworks_compat.js"
    if legacy_jquery_shim.exists():
        errors.append(f"unused compatibility asset remains: {legacy_jquery_shim}")
    for unused_font_pattern in ("fontawesome-webfont.*", "Roboto-Slab-*"):
        unused_fonts = list((static_dir / "css" / "fonts").glob(unused_font_pattern))
        if unused_fonts:
            errors.append(f"superseded theme fonts remain: {unused_fonts}")
    copied_sources = build_dir / "_sources"
    if copied_sources.exists():
        errors.append(f"unreferenced page sources remain: {copied_sources}")


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
