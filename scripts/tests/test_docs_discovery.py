# SPDX-FileCopyrightText: 2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise generated documentation exports without importing AutoLyap."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.docs_discovery import write_discovery_exports
except ModuleNotFoundError:
    write_discovery_exports = None


class DiscoveryExportsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            write_discovery_exports, "The generated-document export helper is missing"
        )
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.outdir = Path(self.temporary_directory.name)

    def write_page(
        self,
        path,
        *,
        canonical,
        body,
        title="Documentation",
        robots="index,follow",
        description="A documented workflow.",
    ):
        target = self.outdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"<html><head><title>{title} | AutoLyap</title>"
            f'<link rel="canonical" href="{canonical}">'
            f'<meta name="robots" content="{robots}">'
            f'<meta name="description" content="{description}"></head>'
            f'<body><nav>Sidebar navigation</nav><div role="main">{body}</div>'
            "<footer>Theme footer</footer></body></html>",
            encoding="utf-8",
        )

    def export(self, *, base_url="https://example.test", macros=None):
        write_discovery_exports(
            self.outdir,
            base_url=base_url,
            project="AutoLyap",
            release="0.4.2",
            description="Computer-assisted Lyapunov analysis.",
            macros=macros or {},
        )

    def test_export_preserves_resolved_article_code_math_and_citations(self):
        # Losing resolved article nodes would omit APIs, equations or references.
        self.write_page(
            "quick_start/index.html",
            canonical="https://example.test/quick_start/",
            title="Quick start",
            body=r"""
<section id="quick-start"><h1>Quick start<a class="headerlink" href="#quick-start"></a></h1>
<p>Use <code>result["status"]</code> and <span class="math">\(x\in\reals\)</span>.</p>
<div class="highlight-python"><div class="highlight"><pre><span class="n">snippet</span> = "```"
print(snippet)
</pre></div></div>
<div class="math">\[\norm{x}^2 \leq 1\]</div>
<dl class="py method"><dt class="sig" id="autolyap.solve"><span class="sig-name">solve</span><span>(</span><em>rho=0.5</em><span>)</span><a class="headerlink" href="#autolyap.solve"></a><a class="reference internal" href="../_modules/autolyap/core/#solve"><span class="viewcode-link">[source]</span></a></dt><dd><p>Return a certificate.</p></dd></dl>
<table><thead><tr><th>Status</th><th>Meaning</th></tr></thead><tbody><tr><td>feasible</td><td>A certificate exists.</td></tr></tbody></table>
<p>See <span id="citation-ref">[<a href="#paper">Paper26</a>]</span>.</p>
<div class="citation" id="paper"><span class="label">[Paper26]</span><p>A. Author. Lyapunov certificates. 2026. <a href="https://doi.org/10.1234/paper">DOI</a>.</p></div>
<script>window.unwanted = true;</script><nav>Article navigation</nav></section>""",
        )
        self.export(
            macros={
                "reals": r"\mathbb{R}",
                "norm": [r"\inner{#1}{#1}^{1/2}", 1],
                "inner": [r"\langle #1, #2\rangle", 2],
                "unused": r"\mathbb{U}",
            }
        )
        result = (self.outdir / "quick_start/index.md").read_text(encoding="utf-8")
        for expected in (
            "Canonical HTML: https://example.test/quick_start/",
            "Documentation version: 0.4.2",
            '<a id="quick-start"></a>',
            '<a id="autolyap.solve"></a>',
            '<a id="citation-ref"></a>',
            '<a id="paper"></a>',
            "# Quick start",
            '````python\nsnippet = "```"\nprint(snippet)\n````',
            r"\(x\in\reals\)",
            r"\[\norm{x}^2 \leq 1\]",
            r"\newcommand{\reals}{\mathbb{R}}",
            r"\newcommand{\norm}[1]{\inner{#1}{#1}^{1/2}}",
            r"\newcommand{\inner}[2]{\langle #1, #2\rangle}",
            "solve(rho=0.5)",
            "Return a certificate.",
            "| Status | Meaning |",
            "| feasible | A certificate exists. |",
            "https://example.test/quick_start/#paper",
            "A. Author. Lyapunov certificates. 2026.",
            "https://doi.org/10.1234/paper",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result)
        for unwanted in (
            "Sidebar navigation",
            "Theme footer",
            "Article navigation",
            "window.unwanted",
            "",
            "[source]",
            r"\newcommand{\unused}",
        ):
            with self.subTest(unwanted=unwanted):
                self.assertNotIn(unwanted, result)

    def test_export_resolves_links_and_images_from_canonical_base_path(self):
        # Relative references must survive fetching the Markdown independently.
        self.write_page(
            "theory/index.html",
            canonical="https://example.test/docs/theory/",
            title="Theory",
            body='<h1>Theory</h1><p><a href="../api/#solve">API</a> '
            '<a href="#equation">Equation</a> <a href="https://other.test/">External</a></p>'
            '<figure id="figure"><a href="../_images/rate.svg"><img src="../_images/rate.svg" alt="Convergence rate"></a><figcaption>Rates over time.</figcaption></figure>',
        )
        self.export(base_url="https://example.test/docs")
        result = (self.outdir / "theory/index.md").read_text(encoding="utf-8")
        for expected in (
            "[API](https://example.test/docs/api/#solve)",
            "[Equation](https://example.test/docs/theory/#equation)",
            "[External](https://other.test/)",
            "![Convergence rate](https://example.test/docs/_images/rate.svg)",
            "Rates over time.",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result)

    def test_nested_autodoc_description_keeps_literal_code_and_tex(self):
        # Markdown definition-list indentation must not rewrite API examples.
        self.write_page(
            "api_reference/index.html",
            canonical="https://example.test/api_reference/",
            body=r"""<h1>API</h1><dl class="py class"><dt class="sig" id="autolyap.Method">Method</dt><dd><dl class="py method"><dt class="sig" id="autolyap.Method.solve">solve()</dt><dd><p>Contraction condition:</p><div class="math">\[\begin{aligned}
x &amp;= y, \\
z &amp;= 0.
\end{aligned}\]</div><div class="highlight-python"><pre>def solve():
    return "certificate"
</pre></div></dd></dl></dd></dl>""",
        )
        self.export()
        result = (self.outdir / "api_reference/index.md").read_text(encoding="utf-8")
        self.assertIn(
            r"\[\begin{aligned}" + "\nx &= y, \\\\\nz &= 0.\n" + r"\end{aligned}\]",
            result,
        )
        self.assertIn('```python\ndef solve():\n    return "certificate"\n```', result)

    def test_noindex_transition_removes_previous_generated_export(self):
        # A page made noindex must not keep its former public Markdown sibling.
        self.write_page(
            "index.html", canonical="https://example.test/", body="<h1>AutoLyap</h1>"
        )
        self.write_page(
            "guide/index.html",
            canonical="https://example.test/guide/",
            body="<h1>Guide</h1>",
        )
        self.export()
        self.assertTrue((self.outdir / "guide/index.md").exists())
        self.write_page(
            "guide/index.html",
            canonical="https://example.test/guide/",
            body="<h1>Guide</h1>",
            robots="noindex,follow",
        )
        self.export()
        self.assertFalse((self.outdir / "guide/index.md").exists())

    def test_nested_list_does_not_indent_literal_code_or_multiline_tex(self):
        # Ancestor list/quote conversion must not modify protected source text.
        self.write_page(
            "theory/index.html",
            canonical="https://example.test/theory/",
            body=r"""<h1>Theory</h1><ol><li><p>The condition is</p><div class="math">\[\begin{aligned}
x &amp;= y, \\
z &amp;= 0.
\end{aligned}\]</div><blockquote><div class="highlight-python"><pre>def solve():
    return "certificate"
</pre></div></blockquote></li></ol>""",
        )
        self.export()
        result = (self.outdir / "theory/index.md").read_text(encoding="utf-8")
        self.assertIn(
            r"\[\begin{aligned}" + "\nx &= y, \\\\\nz &= 0.\n" + r"\end{aligned}\]",
            result,
        )
        self.assertIn('```python\ndef solve():\n    return "certificate"\n```', result)

    def test_equation_export_preserves_both_sphinx_reference_targets(self):
        # Sphinx may place one equation anchor inside the protected math node.
        self.write_page(
            "theory/index.html",
            canonical="https://example.test/theory/",
            body=r"""<h1>Theory</h1><div class="math" id="equation-eq-rate"><span id="eq-rate"></span><span class="eqno">(1)<a class="headerlink" href="#equation-eq-rate"></a></span>\[x \leq y\]</div>""",
        )
        self.export()
        result = (self.outdir / "theory/index.md").read_text(encoding="utf-8")
        self.assertIn('<a id="equation-eq-rate"></a>', result)
        self.assertIn('<a id="eq-rate"></a>', result)
        self.assertIn(r"(1)\[x \leq y\]", result)

    def test_index_contains_only_indexable_exports_with_canonical_version_identity(
        self,
    ):
        # Publishing noindex utility pages would undermine the curated directory.
        for path, route, title in (
            ("index.html", "", "AutoLyap"),
            ("quick_start/index.html", "quick_start/", "Quick start"),
            ("theory/notation/index.html", "theory/notation/", "Notation"),
            ("examples/gradient/index.html", "examples/gradient/", "Gradient method"),
            ("solver_backends/index.html", "solver_backends/", "Solver backends"),
            (
                "iteration_independent_analysis/index.html",
                "iteration_independent_analysis/",
                "Iteration-independent analysis",
            ),
            ("contributing/index.html", "contributing/", "Contributing"),
        ):
            self.write_page(
                path,
                canonical=f"https://example.test/{route}",
                title=title,
                body=f"<h1>{title}</h1><p>Page content.</p>",
            )
        self.write_page(
            "search.html",
            canonical="https://example.test/search.html",
            title="Search",
            robots="noindex,follow",
            body="<h1>Search</h1>",
        )
        self.export()
        index = (self.outdir / "llms.txt").read_text(encoding="utf-8")
        self.assertTrue(index.startswith("# AutoLyap\n"))
        for expected in (
            "> Computer-assisted Lyapunov analysis.",
            "Documentation version: 0.4.2",
            "https://example.test/",
            "https://github.com/AutoLyap/AutoLyap",
            "https://pypi.org/project/autolyap/",
            "## Quick start and guides",
            "## Theory",
            "## Worked examples",
            "## API reference",
            "## Project and development",
            "[Notation](https://example.test/theory/notation/index.md): A documented workflow.",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, index)
        self.assertEqual(len(list(self.outdir.rglob("*.md"))), 7)
        self.assertEqual(index.count("https://example.test/quick_start/index.md"), 1)
        self.assertNotIn("search", index)
        self.assertFalse((self.outdir / "search.md").exists())

    def test_indexable_missing_canonical_or_article_fails_loudly(self):
        # An incomplete export must not silently replace documentation.
        for markup in (
            '<html><body><div role="main"><h1>Page</h1></div></body></html>',
            '<html><head><link rel="canonical" href="https://example.test/"></head><body><nav>Only navigation</nav></body></html>',
        ):
            with self.subTest(markup=markup):
                (self.outdir / "index.html").write_text(markup, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.export()


if __name__ == "__main__":
    unittest.main()
