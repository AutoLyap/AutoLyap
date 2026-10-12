# SPDX-FileCopyrightText: 2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Audit technical SEO invariants in a Sphinx ``dirhtml`` build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_BASE_URL = "https://autolyap.github.io"
REPOSITORY_URL = "https://github.com/AutoLyap/AutoLyap"
PACKAGE_URL = "https://pypi.org/project/autolyap/"
SITEMAP_NAMESPACE = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
CRITICAL_INDEXABLE_PATHS = {
    "index.html",
    "quick_start/index.html",
    "theory/index.html",
    "examples/index.html",
    "api_reference/index.html",
    "base_algorithms/index.html",
    "iteration_independent_analysis/index.html",
    "iteration_dependent_analysis/index.html",
}
PAGE_SCHEMA_TYPES = {
    "APIReference",
    "CollectionPage",
    "LearningResource",
    "TechArticle",
    "WebPage",
}


@dataclass
class PageHead:
    """SEO-relevant values extracted from one generated HTML document."""

    title_parts: list[str] = field(default_factory=list)
    meta_names: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    meta_properties: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    canonical_urls: list[str] = field(default_factory=list)
    markdown_urls: list[str] = field(default_factory=list)
    describedby_urls: list[str] = field(default_factory=list)
    json_ld_texts: list[str] = field(default_factory=list)
    html_languages: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return " ".join("".join(self.title_parts).split())


class _HeadParser(HTMLParser):
    """Collect head metadata without requiring third-party HTML packages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.result = PageHead()
        self._in_head = False
        self._in_title = False
        self._json_ld_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {str(key).lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "head":
            self._in_head = True
        elif tag == "html" and attributes.get("lang"):
            self.result.html_languages.append(attributes["lang"])
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            content = attributes.get("content", "").strip()
            if attributes.get("name"):
                self.result.meta_names[attributes["name"].lower()].append(content)
            if attributes.get("property"):
                self.result.meta_properties[attributes["property"].lower()].append(
                    content
                )
        elif tag == "link":
            relations = attributes.get("rel", "").lower().split()
            href = attributes.get("href", "").strip()
            if "canonical" in relations:
                self.result.canonical_urls.append(href)
            if self._in_head:
                if "alternate" in relations and attributes.get("type", "").lower() == (
                    "text/markdown"
                ):
                    self.result.markdown_urls.append(href)
                if "describedby" in relations:
                    self.result.describedby_urls.append(href)
        elif tag == "script" and attributes.get("type", "").lower() == (
            "application/ld+json"
        ):
            self._json_ld_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "head":
            self._in_head = False
        elif tag == "title":
            self._in_title = False
        elif tag == "script" and self._json_ld_parts is not None:
            self.result.json_ld_texts.append("".join(self._json_ld_parts).strip())
            self._json_ld_parts = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.result.title_parts.append(data)
        if self._json_ld_parts is not None:
            self._json_ld_parts.append(data)


def _parse_page(path: Path) -> tuple[str, PageHead]:
    text = path.read_text(encoding="utf-8")
    parser = _HeadParser()
    parser.feed(text)
    parser.close()
    return text, parser.result


def _expected_url(path: Path, output_dir: Path, base_url: str) -> str:
    relative = path.relative_to(output_dir)
    if relative.name == "index.html":
        route = relative.parent.as_posix().strip(".")
        return f"{base_url}/{route + '/' if route else ''}"
    return f"{base_url}/{relative.as_posix()}"


def _url_output_path(url: str, output_dir: Path, base_url: str) -> Path | None:
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    if (parsed_url.scheme, parsed_url.netloc) != (
        parsed_base.scheme,
        parsed_base.netloc,
    ):
        return None

    base_path = parsed_base.path.rstrip("/")
    url_path = unquote(parsed_url.path)
    if base_path and not url_path.startswith(f"{base_path}/") and url_path != base_path:
        return None
    relative_path = url_path[len(base_path) :].lstrip("/")
    if not relative_path:
        return output_dir / "index.html"
    if url_path.endswith("/"):
        return output_dir / relative_path / "index.html"
    return output_dir / relative_path


def _one(values: list[str], label: str, page: str, errors: list[str]) -> str:
    if len(values) != 1 or not values[0]:
        errors.append(f"{page}: expected one nonempty {label}, got {values!r}")
        return ""
    return values[0]


def _schema_types(document: dict) -> set[str]:
    value = document.get("@type", [])
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _require_schema_reference(
    document: dict, field_name: str, expected_id: str, page: str, errors: list[str]
) -> None:
    value = document.get(field_name)
    actual_id = value.get("@id") if isinstance(value, dict) else None
    if actual_id != expected_id:
        errors.append(
            f"{page}: schema {field_name} must reference {expected_id!r}, "
            f"got {actual_id!r}"
        )


def _markdown_url(page_url: str) -> str:
    if page_url.endswith("/"):
        return f"{page_url}index.md"
    return f"{page_url.removesuffix('.html')}.md"


def _audit_agent_discovery(
    output_dir: Path, base_url: str, pages: list[dict], version: str, errors: list[str]
) -> None:
    expected_markdown_urls = set()
    for page in pages:
        relative = page["relative"]
        export_path = (output_dir / relative).with_suffix(".md")
        head = page["head"]
        if page["noindex"]:
            if head.markdown_urls or export_path.is_file():
                errors.append(
                    f"{relative}: noindex page must not expose a Markdown export"
                )
            if head.describedby_urls:
                errors.append(
                    f"{relative}: noindex page must not expose discovery links"
                )
            continue

        expected_url = _expected_url(output_dir / relative, output_dir, base_url)
        expected_markdown = _markdown_url(expected_url)
        expected_markdown_urls.add(expected_markdown)
        markdown_url = _one(
            head.markdown_urls, "Markdown alternate link", relative, errors
        )
        if markdown_url and markdown_url != expected_markdown:
            errors.append(
                f"{relative}: Markdown alternate {markdown_url!r} != {expected_markdown!r}"
            )
        describedby = _one(
            head.describedby_urls, "llms.txt describedby link", relative, errors
        )
        if describedby and describedby != f"{base_url}/llms.txt":
            errors.append(f"{relative}: describedby must point to {base_url}/llms.txt")

        if not export_path.is_file():
            errors.append(f"{relative}: missing Markdown export {export_path.name!r}")
            continue
        export_text = export_path.read_text(encoding="utf-8")
        export_lines = export_text.splitlines()
        header_lines = export_lines[:10]
        canonical_header = f"Canonical HTML: {expected_url}"
        version_header = f"Documentation version: {version}"
        if export_lines[:1] != [canonical_header] or [
            line for line in header_lines if line.startswith("Canonical HTML:")
        ] != [canonical_header]:
            errors.append(
                f"{relative}: Markdown export lacks correct canonical HTML header"
            )
        if (
            not version
            or export_lines[1:2] != [version_header]
            or [
                line
                for line in header_lines
                if line.startswith("Documentation version:")
            ]
            != [version_header]
        ):
            errors.append(
                f"{relative}: Markdown export lacks matching documentation version"
            )
        content_lines = [
            line
            for line in export_lines
            if not line.startswith(("Canonical HTML:", "Documentation version:"))
        ]
        if not "\n".join(content_lines).strip():
            errors.append(f"{relative}: Markdown export has no documentation content")

    index_path = output_dir / "llms.txt"
    if not index_path.is_file():
        errors.append("Missing llms.txt")
        return
    index_text = index_path.read_text(encoding="utf-8")
    if not index_text.startswith("# AutoLyap\n"):
        errors.append("llms.txt lacks the AutoLyap heading")
    version_lines = [
        line
        for line in index_text.splitlines()
        if line.startswith("Documentation version:")
    ]
    if not version or version_lines != [f"Documentation version: {version}"]:
        errors.append("llms.txt lacks matching documentation version")
    listed_urls = [
        url
        for url in re.findall(r"\[[^\n]*?\]\((https?://[^)\s]+)\)", index_text)
        if url.endswith(".md")
    ]
    for url in sorted(expected_markdown_urls):
        count = listed_urls.count(url)
        if count != 1:
            errors.append(
                f"llms.txt: expected one Markdown entry for {url!r}, got {count}"
            )
    unexpected = set(listed_urls) - expected_markdown_urls
    if unexpected:
        errors.append(f"llms.txt: unexpected Markdown exports: {sorted(unexpected)!r}")


def _valid_iso_datetime(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})",
            value,
        )
    )


def _body_hash(text: str) -> str:
    match = re.search(r"<body\b.*?</body>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return ""
    return hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()


def audit(
    output_dir: Path,
    *,
    base_url: str,
    baseline_dir: Path | None = None,
) -> list[str]:
    """Return all SEO audit errors for ``output_dir``."""
    errors: list[str] = []
    pages: list[dict] = []
    software_version = ""
    output_dir = output_dir.resolve()
    base_url = base_url.rstrip("/")

    html_paths = sorted(output_dir.rglob("*.html"))
    if not html_paths:
        return [f"No HTML files found under {output_dir}"]

    for path in html_paths:
        relative = path.relative_to(output_dir).as_posix()
        text, head = _parse_page(path)
        expected_url = _expected_url(path, output_dir, base_url)
        description = _one(
            head.meta_names.get("description", []), "meta description", relative, errors
        )
        robots = _one(
            head.meta_names.get("robots", []), "robots meta", relative, errors
        )
        canonical = _one(head.canonical_urls, "canonical URL", relative, errors)
        og_url = _one(
            head.meta_properties.get("og:url", []), "Open Graph URL", relative, errors
        )
        twitter_url = _one(
            head.meta_names.get("twitter:url", []), "Twitter URL", relative, errors
        )

        if not head.title:
            errors.append(f"{relative}: missing title")
        if head.html_languages != ["en"]:
            errors.append(
                f"{relative}: expected one html lang='en', got {head.html_languages!r}"
            )
        for label, value in (
            ("canonical", canonical),
            ("og:url", og_url),
            ("twitter:url", twitter_url),
        ):
            if value and value != expected_url:
                errors.append(f"{relative}: {label} {value!r} != {expected_url!r}")

        required_names = (
            "author",
            "viewport",
            "twitter:card",
            "twitter:title",
            "twitter:description",
            "twitter:image",
            "twitter:image:alt",
        )
        required_properties = (
            "og:type",
            "og:site_name",
            "og:title",
            "og:description",
            "og:image",
            "og:image:alt",
        )
        for name in required_names:
            _one(head.meta_names.get(name, []), f"meta name={name}", relative, errors)
        for property_name in required_properties:
            _one(
                head.meta_properties.get(property_name, []),
                f"meta property={property_name}",
                relative,
                errors,
            )

        for label, image_url in (
            ("og:image", head.meta_properties.get("og:image", [""])[0]),
            ("twitter:image", head.meta_names.get("twitter:image", [""])[0]),
        ):
            if image_url and not urlparse(image_url).scheme:
                errors.append(f"{relative}: {label} must be absolute: {image_url!r}")

        robot_tokens = {token.strip().lower() for token in robots.split(",")}
        noindex = "noindex" in robot_tokens
        if relative in CRITICAL_INDEXABLE_PATHS and (
            noindex or "none" in robot_tokens or "nofollow" in robot_tokens
        ):
            errors.append(
                f"{relative}: critical page must remain indexable and followable"
            )
        if not noindex and "index" not in robot_tokens:
            errors.append(f"{relative}: indexable page lacks an index robots directive")
        if not noindex and (
            "follow" not in robot_tokens or {"none", "nofollow"} & robot_tokens
        ):
            errors.append(
                f"{relative}: indexable page lacks a valid follow robots directive"
            )
        if not noindex and not 70 <= len(description) <= 160:
            errors.append(
                f"{relative}: meta description length {len(description)} is outside 70..160"
            )
        if not noindex and len(head.title) > 65:
            errors.append(
                f"{relative}: title is longer than 65 characters: {head.title!r}"
            )

        json_documents: list[dict] = []
        for position, json_text in enumerate(head.json_ld_texts, start=1):
            try:
                document = json.loads(json_text)
            except json.JSONDecodeError as exc:
                errors.append(f"{relative}: JSON-LD block {position} is invalid: {exc}")
                continue
            if not isinstance(document, dict):
                errors.append(f"{relative}: JSON-LD block {position} is not an object")
                continue
            json_documents.append(document)

        page_schemas = [
            document
            for document in json_documents
            if _schema_types(document) & PAGE_SCHEMA_TYPES
            and document.get("url") == canonical
        ]
        if len(page_schemas) != 1:
            errors.append(
                f"{relative}: expected one page schema for canonical, got {len(page_schemas)}"
            )
        else:
            page_schema = page_schemas[0]
            if page_schema.get("@id") != f"{expected_url}#webpage":
                errors.append(f"{relative}: page schema has an invalid @id")
            for field_name, expected_id in (
                ("isPartOf", f"{base_url}/#website"),
                ("about", f"{base_url}/#software"),
                ("author", f"{base_url}/#organization"),
                ("publisher", f"{base_url}/#organization"),
            ):
                _require_schema_reference(
                    page_schema, field_name, expected_id, relative, errors
                )
            for date_field in ("datePublished", "dateModified"):
                if date_field not in page_schema:
                    continue
                date_value = page_schema[date_field]
                if not isinstance(date_value, str) or not _valid_iso_datetime(
                    date_value
                ):
                    errors.append(
                        f"{relative}: invalid schema {date_field}: {date_value!r}"
                    )
            if not noindex:
                if relative != "index.html":
                    _require_schema_reference(
                        page_schema,
                        "breadcrumb",
                        f"{expected_url}#breadcrumb",
                        relative,
                        errors,
                    )

        breadcrumbs = [
            document
            for document in json_documents
            if "BreadcrumbList" in _schema_types(document)
        ]
        if not noindex and relative != "index.html" and len(breadcrumbs) != 1:
            errors.append(
                f"{relative}: expected one BreadcrumbList, got {len(breadcrumbs)}"
            )
        for breadcrumb in breadcrumbs:
            if breadcrumb.get("@id") != f"{expected_url}#breadcrumb":
                errors.append(f"{relative}: BreadcrumbList has an invalid @id")
            items = breadcrumb.get("itemListElement", [])
            positions = [
                item.get("position") for item in items if isinstance(item, dict)
            ]
            urls = [item.get("item") for item in items if isinstance(item, dict)]
            if positions != list(range(1, len(items) + 1)):
                errors.append(f"{relative}: breadcrumb positions are not consecutive")
            if not urls or urls[0] != f"{base_url}/" or urls[-1] != canonical:
                errors.append(f"{relative}: breadcrumb endpoints are invalid: {urls!r}")
            if len(urls) != len(set(urls)):
                errors.append(f"{relative}: breadcrumb URLs are duplicated: {urls!r}")
            for item in items:
                if not isinstance(item, dict) or not item.get("name"):
                    errors.append(f"{relative}: breadcrumb item lacks a name")
                    continue
                target = _url_output_path(
                    str(item.get("item", "")), output_dir, base_url
                )
                if target is None or not target.is_file():
                    errors.append(
                        f"{relative}: breadcrumb URL does not resolve: {item.get('item')!r}"
                    )

        entity_types = [_schema_types(document) for document in json_documents]
        site_entities = sum("WebSite" in types for types in entity_types)
        organization_entities = sum("Organization" in types for types in entity_types)
        source_entities = sum("SoftwareSourceCode" in types for types in entity_types)
        expected_site_entities = 1 if relative == "index.html" else 0
        if (site_entities, organization_entities, source_entities) != (
            expected_site_entities,
            expected_site_entities,
            expected_site_entities,
        ):
            errors.append(
                f"{relative}: unexpected site entity counts "
                f"{(site_entities, organization_entities, source_entities)!r}"
            )
        if relative == "index.html":
            for document in json_documents:
                types = _schema_types(document)
                for entity_type, expected_id in (
                    ("WebSite", f"{base_url}/#website"),
                    ("Organization", f"{base_url}/#organization"),
                    ("SoftwareSourceCode", f"{base_url}/#software"),
                ):
                    if entity_type not in types:
                        continue
                    if document.get("@id") != expected_id:
                        errors.append(f"{relative}: {entity_type} has an invalid @id")
                    if document.get("url") != f"{base_url}/":
                        errors.append(
                            f"{relative}: {entity_type} has an invalid documentation URL"
                        )
                    if entity_type != "Organization":
                        _require_schema_reference(
                            document,
                            "publisher",
                            f"{base_url}/#organization",
                            relative,
                            errors,
                        )
                if "SoftwareSourceCode" in types:
                    value = document.get("version")
                    if not isinstance(value, str) or not value.strip():
                        errors.append(f"{relative}: SoftwareSourceCode lacks a version")
                    else:
                        software_version = value
                    if document.get("codeRepository") != REPOSITORY_URL:
                        errors.append(
                            f"{relative}: SoftwareSourceCode has an invalid codeRepository"
                        )
                    identities = document.get("sameAs", [])
                    if (
                        not isinstance(identities, list)
                        or not all(isinstance(identity, str) for identity in identities)
                        or not {REPOSITORY_URL, PACKAGE_URL}.issubset(identities)
                    ):
                        errors.append(
                            f"{relative}: SoftwareSourceCode lacks repository/package identities"
                        )

        if baseline_dir is not None:
            baseline_path = baseline_dir.resolve() / path.relative_to(output_dir)
            if not baseline_path.is_file():
                errors.append(f"{relative}: missing from body-comparison baseline")
            else:
                baseline_text = baseline_path.read_text(encoding="utf-8")
                if _body_hash(text) != _body_hash(baseline_text):
                    errors.append(f"{relative}: rendered <body> changed")

        pages.append(
            {
                "relative": relative,
                "title": head.title,
                "description": description,
                "canonical": canonical,
                "noindex": noindex,
                "head": head,
            }
        )

    indexable_pages = [page for page in pages if not page["noindex"]]
    missing_critical = CRITICAL_INDEXABLE_PATHS - {page["relative"] for page in pages}
    if missing_critical:
        errors.append(f"Missing critical pages: {sorted(missing_critical)!r}")
    _audit_agent_discovery(output_dir, base_url, pages, software_version, errors)
    for field_name in ("title", "description", "canonical"):
        occurrences: dict[str, list[str]] = defaultdict(list)
        for page in indexable_pages:
            occurrences[page[field_name]].append(page["relative"])
        for value, relative_paths in occurrences.items():
            if not value or len(relative_paths) > 1:
                errors.append(
                    f"Indexable {field_name} is empty or duplicated: "
                    f"{value!r} on {relative_paths!r}"
                )

    sitemap_path = output_dir / "sitemap.xml"
    if not sitemap_path.is_file():
        errors.append("Missing sitemap.xml")
    else:
        try:
            sitemap_root = ET.parse(sitemap_path).getroot()
        except ET.ParseError as exc:
            errors.append(f"sitemap.xml is invalid XML: {exc}")
        else:
            sitemap_entries = sitemap_root.findall("s:url", SITEMAP_NAMESPACE)
            sitemap_urls: list[str] = []
            for entry in sitemap_entries:
                location = entry.findtext(
                    "s:loc", default="", namespaces=SITEMAP_NAMESPACE
                )
                lastmod = entry.findtext(
                    "s:lastmod", default="", namespaces=SITEMAP_NAMESPACE
                )
                sitemap_urls.append(location)
                if entry.find("s:lastmod", SITEMAP_NAMESPACE) is not None and not (
                    _valid_iso_datetime(lastmod)
                ):
                    errors.append(
                        f"sitemap.xml: invalid lastmod for {location!r}: {lastmod!r}"
                    )
                if entry.find("s:priority", SITEMAP_NAMESPACE) is not None:
                    errors.append(
                        f"sitemap.xml: priority must be omitted for {location!r}"
                    )
                if entry.find("s:changefreq", SITEMAP_NAMESPACE) is not None:
                    errors.append(
                        f"sitemap.xml: changefreq must be omitted for {location!r}"
                    )
                target = _url_output_path(location, output_dir, base_url)
                if target is None or not target.is_file():
                    errors.append(f"sitemap.xml: URL does not resolve: {location!r}")

            expected_sitemap_urls = {page["canonical"] for page in indexable_pages}
            actual_sitemap_urls = set(sitemap_urls)
            if len(sitemap_urls) != len(actual_sitemap_urls):
                errors.append("sitemap.xml contains duplicate URLs")
            if actual_sitemap_urls != expected_sitemap_urls:
                errors.append(
                    "sitemap.xml indexability mismatch: "
                    f"missing={sorted(expected_sitemap_urls - actual_sitemap_urls)!r}, "
                    f"extra={sorted(actual_sitemap_urls - expected_sitemap_urls)!r}"
                )

    robots_path = output_dir / "robots.txt"
    if not robots_path.is_file():
        errors.append("Missing robots.txt")
    else:
        robots_text = robots_path.read_text(encoding="utf-8")
        expected_sitemap_line = f"Sitemap: {base_url}/sitemap.xml"
        if expected_sitemap_line not in robots_text.splitlines():
            errors.append(f"robots.txt lacks {expected_sitemap_line!r}")
        disallow_lines = [
            line.strip()
            for line in robots_text.splitlines()
            if line.strip().lower().startswith("disallow:")
        ]
        if disallow_lines != ["Disallow: /_sources/"]:
            errors.append(
                f"robots.txt must only disallow raw sources; got {disallow_lines!r}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Sphinx dirhtml output directory")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Optional dirhtml build whose rendered bodies must remain identical",
    )
    arguments = parser.parse_args()

    errors = audit(
        arguments.output_dir,
        base_url=arguments.base_url,
        baseline_dir=arguments.baseline_dir,
    )
    if errors:
        print(f"SEO audit failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    page_count = len(list(arguments.output_dir.rglob("*.html")))
    print(f"SEO audit passed for {page_count} generated HTML pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
