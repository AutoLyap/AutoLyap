# SPDX-FileCopyrightText: 2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Test SEO regressions against a built documentation site without core imports."""

from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

seo = importlib.import_module("scripts.audit_docs_seo")

BASE = seo.DEFAULT_BASE_URL


class DocumentationSEOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "docs/build/dirhtml"
        if not (source / "index.html").is_file():
            raise RuntimeError(
                "Build the dirhtml documentation before running SEO tests"
            )
        cls.fixture = {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
            and (
                path.suffix in {".html", ".md"}
                or path.name in {"sitemap.xml", "robots.txt", "llms.txt"}
            )
        }
        _, head = seo._parse_page(source / "index.html")
        cls.version = next(
            document["version"]
            for text in head.json_ld_texts
            if "SoftwareSourceCode" in seo._schema_types(document := json.loads(text))
        )

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory(prefix="autolyap-seo-test-")
        self.addCleanup(temporary_directory.cleanup)
        self.out = Path(temporary_directory.name)
        for relative, content in self.fixture.items():
            path = self.out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def errors(self):
        return seo.audit(self.out, base_url=BASE)

    def update_schema(self, relative, schema_type, update):
        path = self.out / relative
        markup, head = seo._parse_page(path)
        matches = [
            (text, document)
            for text in head.json_ld_texts
            if schema_type in seo._schema_types(document := json.loads(text))
        ]
        self.assertEqual(len(matches), 1)
        text, document = matches[0]
        update(document)
        path.write_text(markup.replace(text, json.dumps(document)), encoding="utf-8")

    def sitemap_entry(self, route="quick_start/"):
        tree = ET.parse(self.out / "sitemap.xml")
        entry = next(
            entry
            for entry in tree.getroot().findall("s:url", seo.SITEMAP_NAMESPACE)
            if entry.findtext("s:loc", namespaces=seo.SITEMAP_NAMESPACE)
            == BASE + "/" + route
        )
        return tree, entry

    def test_complete_built_site_passes(self):
        self.assertEqual(self.errors(), [])

    def test_critical_page_cannot_be_noindexed_with_matching_sitemap(self):
        path = self.out / "quick_start/index.html"
        markup = re.sub(
            r'(<meta name="robots" content=")[^"]*',
            r"\1noindex,follow",
            path.read_text(encoding="utf-8"),
        )
        markup = re.sub(r'<link rel="(?:alternate|describedby)"[^>]*>\n?', "", markup)
        path.write_text(markup, encoding="utf-8")
        tree, entry = self.sitemap_entry()
        tree.getroot().remove(entry)
        tree.write(self.out / "sitemap.xml", encoding="utf-8")
        (self.out / "quick_start/index.md").unlink()
        index = self.out / "llms.txt"
        index.write_text(
            "\n".join(
                line
                for line in index.read_text().splitlines()
                if BASE + "/quick_start/" not in line
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertTrue(
            any("critical page" in error.lower() for error in self.errors())
        )

    def test_dangling_site_entity_relation_is_rejected(self):
        self.update_schema(
            "quick_start/index.html",
            "TechArticle",
            lambda document: document.update(
                isPartOf={"@id": BASE + "/#missing-website"}
            ),
        )
        self.assertTrue(any("isPartOf" in error for error in self.errors()))

    def test_missing_markdown_export_is_rejected(self):
        (self.out / "quick_start/index.md").unlink()
        self.assertTrue(
            any("missing Markdown export" in error for error in self.errors())
        )

    def test_missing_markdown_discovery_link_is_rejected(self):
        path = self.out / "quick_start/index.html"
        path.write_text(
            re.sub(r'<link rel="alternate"[^>]*>\n?', "", path.read_text()),
            encoding="utf-8",
        )
        self.assertTrue(any("Markdown alternate" in error for error in self.errors()))

    def test_missing_llms_entry_is_rejected(self):
        index = self.out / "llms.txt"
        index.write_text(
            "\n".join(
                line
                for line in index.read_text().splitlines()
                if BASE + "/quick_start/index.md" not in line
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertTrue(
            any(
                "llms.txt" in error and "quick_start" in error
                for error in self.errors()
            )
        )

    def test_incorrect_markdown_canonical_is_rejected(self):
        path = self.out / "quick_start/index.md"
        path.write_text(
            path.read_text().replace(BASE + "/quick_start/", BASE + "/wrong-page/"),
            encoding="utf-8",
        )
        self.assertTrue(any("canonical HTML" in error for error in self.errors()))

    def test_conflicting_export_metadata_is_rejected(self):
        path = self.out / "quick_start/index.md"
        path.write_text(
            f"Canonical HTML: {BASE}/wrong/\nDocumentation version: invalid\n"
            + path.read_text(),
            encoding="utf-8",
        )
        self.assertTrue(any("canonical HTML" in error for error in self.errors()))

    def test_conflicting_llms_version_is_rejected(self):
        path = self.out / "llms.txt"
        header = f"Documentation version: {self.version}"
        path.write_text(
            path.read_text().replace(
                header, "Documentation version: invalid\n" + header
            ),
            encoding="utf-8",
        )
        self.assertTrue(
            any(
                "llms.txt lacks matching documentation version" in error
                for error in self.errors()
            )
        )

    def test_inline_duplicate_and_noindex_exports_are_rejected(self):
        path = self.out / "llms.txt"
        path.write_text(
            path.read_text()
            + f"- Extra exports: [Quick start]({BASE}/quick_start/index.md), [Search]({BASE}/search/index.md)\n",
            encoding="utf-8",
        )
        errors = self.errors()
        self.assertTrue(
            any(
                "got 2" in error and "quick_start/index.md" in error for error in errors
            )
        )
        self.assertTrue(
            any(
                "unexpected Markdown exports" in error and "search/index.md" in error
                for error in errors
            )
        )

    def test_malformed_software_identity_reports_an_error(self):
        self.update_schema(
            "index.html",
            "SoftwareSourceCode",
            lambda document: document.update(sameAs=[{"@id": seo.REPOSITORY_URL}]),
        )
        self.assertTrue(any("identities" in error for error in self.errors()))

    def test_noindex_page_cannot_advertise_an_export(self):
        path = self.out / "search/index.html"
        path.write_text(
            path.read_text().replace(
                "</head>",
                f'<link rel="alternate" type="text/markdown" href="{BASE}/search/index.md"/>\n</head>',
            ),
            encoding="utf-8",
        )
        self.assertTrue(
            any(
                "noindex page" in error and "Markdown" in error
                for error in self.errors()
            )
        )

    def test_unknown_source_dates_can_be_omitted(self):
        def remove_dates(document):
            document.pop("datePublished", None)
            document.pop("dateModified", None)

        self.update_schema("quick_start/index.html", "TechArticle", remove_dates)
        tree, entry = self.sitemap_entry()
        lastmod = entry.find("s:lastmod", seo.SITEMAP_NAMESPACE)
        if lastmod is not None:
            entry.remove(lastmod)
        tree.write(self.out / "sitemap.xml", encoding="utf-8")
        self.assertEqual(self.errors(), [])

    def test_emitted_schema_dates_must_be_nonempty_iso_values(self):
        path = self.out / "quick_start/index.html"
        original = path.read_text(encoding="utf-8")
        for value in ("", None, "not-a-date"):
            with self.subTest(value=value):
                path.write_text(original, encoding="utf-8")
                self.update_schema(
                    "quick_start/index.html",
                    "TechArticle",
                    lambda document: document.update(
                        datePublished=value, dateModified=value
                    ),
                )
                errors = self.errors()
                for field in ("datePublished", "dateModified"):
                    self.assertTrue(any("schema " + field in error for error in errors))

    def test_emitted_sitemap_dates_must_be_nonempty_iso_values(self):
        for value in ("", "not-a-date"):
            with self.subTest(value=value):
                tree, entry = self.sitemap_entry()
                lastmod = entry.find("s:lastmod", seo.SITEMAP_NAMESPACE)
                if lastmod is None:
                    lastmod = ET.SubElement(
                        entry, "{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod"
                    )
                lastmod.text = value
                tree.write(self.out / "sitemap.xml", encoding="utf-8")
                self.assertTrue(
                    any(
                        "sitemap.xml: invalid lastmod" in error
                        for error in self.errors()
                    )
                )


if __name__ == "__main__":
    unittest.main()
