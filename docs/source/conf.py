import os
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from sphinx import addnodes
from sphinx.domains.python._object import PyObject
project = "AutoLyap"
author = "Manu Upadhyaya"
copyright = f"{date.today().year}, Manu Upadhyaya"

root = Path(__file__).resolve().parents[2]
release = (root / "VERSION").read_text(encoding="utf-8").strip()
version = release

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxcontrib.bibtex",
]

autodoc_mock_imports = [
    "cvxpy",
    "mosek",
    "mosek.fusion",
    "mosek.fusion.pythonic",
]

autodoc_type_aliases = {
    "CacheValueT": "typing.Any",
}

templates_path = ["_templates"]
exclude_patterns = ["release_notes/_template.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["custom.css"]
html_baseurl = "https://autolyap.github.io/"
html_context = {
    "seo_site_name": project,
    "seo_site_description": (
        "AutoLyap is a Python package for computer-assisted Lyapunov analyses "
        "of first-order optimization and inclusion methods."
    ),
    "seo_default_keywords": [
        "AutoLyap",
        "Lyapunov analysis Python",
        "first-order optimization",
        "semidefinite programming",
        "convergence analysis",
    ],
    "seo_pages": {
        "index": {
            "description": (
                "AutoLyap is a Python package for computer-assisted Lyapunov "
                "analyses of first-order optimization and inclusion methods."
            ),
            "keywords": [
                "AutoLyap",
                "Lyapunov analysis Python",
                "first-order optimization convergence analysis",
                "semidefinite programming for optimization",
            ],
        },
        "quick_start": {
            "description": (
                "Quick start guide for AutoLyap with iteration-independent and "
                "iteration-dependent Lyapunov analysis workflows."
            ),
            "keywords": [
                "AutoLyap quick start",
                "iteration-independent Lyapunov analysis",
                "iteration-dependent Lyapunov analysis",
                "bisection search rho",
            ],
        },
        "api_reference": {
            "description": (
                "AutoLyap API reference for algorithms, problem classes, and "
                "Lyapunov analysis helpers."
            ),
            "keywords": [
                "AutoLyap API",
                "Lyapunov analysis API",
                "optimization algorithms Python",
            ],
        },
        "algorithms": {
            "description": (
                "Overview of algorithm abstractions and concrete first-order "
                "methods supported in AutoLyap."
            ),
            "keywords": [
                "AutoLyap algorithms",
                "first-order methods",
                "optimization algorithm analysis",
            ],
        },
        "base_algorithms": {
            "description": (
                "Base algorithm interfaces in AutoLyap for defining iterative "
                "optimization and inclusion methods."
            ),
            "keywords": [
                "AutoLyap base algorithms",
                "algorithm interface Python",
                "iterative method abstraction",
            ],
        },
        "concrete_algorithms": {
            "description": (
                "Concrete algorithm implementations in AutoLyap, including "
                "gradient, proximal, heavy-ball, and accelerated methods."
            ),
            "keywords": [
                "AutoLyap concrete algorithms",
                "gradient method analysis",
                "proximal and accelerated methods",
            ],
        },
        "examples": {
            "description": (
                "AutoLyap examples for proximal point, proximal gradient, and "
                "heavy-ball analyses."
            ),
            "keywords": [
                "AutoLyap examples",
                "proximal point method analysis",
                "proximal gradient Lyapunov analysis",
                "heavy-ball method convergence",
            ],
        },
        "examples/proximal_point": {
            "description": (
                "Proximal point example in AutoLyap with a computer-assisted "
                "Lyapunov convergence analysis."
            ),
            "keywords": [
                "proximal point Lyapunov analysis",
                "AutoLyap proximal point",
                "first-order method convergence proof",
            ],
        },
        "examples/proximal_gradient": {
            "description": (
                "Proximal gradient example in AutoLyap with SDP-based Lyapunov "
                "verification."
            ),
            "keywords": [
                "proximal gradient Lyapunov analysis",
                "AutoLyap proximal gradient",
                "SDP convergence analysis",
            ],
        },
        "examples/heavy_ball": {
            "description": (
                "Heavy-ball example in AutoLyap showing certified parameter "
                "regions for smooth convex optimization."
            ),
            "keywords": [
                "heavy-ball method analysis",
                "AutoLyap heavy-ball",
                "smooth convex optimization convergence",
            ],
        },
        "function_classes": {
            "description": (
                "Function interpolation classes in AutoLyap for modeling convex, "
                "smooth, and strongly convex objectives."
            ),
            "keywords": [
                "AutoLyap function classes",
                "convex and smooth interpolation",
                "optimization problem modeling",
            ],
        },
        "operator_classes": {
            "description": (
                "Operator interpolation classes in AutoLyap for monotone, "
                "Lipschitz, and cocoercive operator models."
            ),
            "keywords": [
                "AutoLyap operator classes",
                "monotone operator analysis",
                "cocoercive and Lipschitz operators",
            ],
        },
        "problem_class": {
            "description": (
                "Problem class definitions in AutoLyap for constructing "
                "optimization and inclusion formulations."
            ),
            "keywords": [
                "AutoLyap problem class",
                "inclusion problem modeling",
                "interpolation indices",
            ],
        },
        "iteration_independent_analysis": {
            "description": (
                "Iteration-independent Lyapunov analysis tools in AutoLyap for "
                "linear and sublinear convergence certification."
            ),
            "keywords": [
                "iteration-independent Lyapunov analysis",
                "linear convergence certificate",
                "AutoLyap iteration independent",
            ],
        },
        "iteration_dependent_analysis": {
            "description": (
                "Iteration-dependent Lyapunov analysis tools in AutoLyap for "
                "finite-horizon and chained-inequality certification."
            ),
            "keywords": [
                "iteration-dependent Lyapunov analysis",
                "finite-horizon convergence",
                "AutoLyap iteration dependent",
            ],
        },
        "lyapunov_analyses": {
            "description": (
                "Lyapunov analysis entry points in AutoLyap for constructing and "
                "verifying convergence certificates."
            ),
            "keywords": [
                "AutoLyap Lyapunov analyses",
                "convergence certificate verification",
                "SDP Lyapunov methods",
            ],
        },
        "solver_backends": {
            "description": (
                "Solver backend options in AutoLyap, including MOSEK Fusion and "
                "CVXPY-based workflows."
            ),
            "keywords": [
                "AutoLyap solver backends",
                "MOSEK Fusion CVXPY",
                "SDP solver configuration",
            ],
        },
        "contributing": {
            "description": (
                "Contribution guidelines for AutoLyap development, testing, and "
                "documentation workflows."
            ),
            "keywords": [
                "contribute to AutoLyap",
                "AutoLyap development guide",
                "testing and documentation workflow",
            ],
        },
        "whats_new": {
            "description": "Release highlights and feature updates for AutoLyap.",
            "keywords": [
                "AutoLyap release notes",
                "AutoLyap changelog",
                "AutoLyap v0.2.0",
            ],
        },
        "release_notes/v0_2_0": {
            "description": (
                "AutoLyap v0.2.0 release notes with new diagnostics, verbosity "
                "output improvements, and documentation updates."
            ),
            "keywords": [
                "AutoLyap v0.2.0",
                "Lyapunov diagnostics",
                "SDP constraint diagnostics",
            ],
        },
    },
    "seo_repo_url": "https://github.com/AutoLyap/AutoLyap",
    "seo_baseurl": "https://autolyap.github.io",
    "seo_author": author,
    "seo_in_language": "en-US",
}
maximum_signature_line_length = 1
toc_object_entries = True

# MathJax macros aligned with Paper/ver_5/commands.tex (subset used in docs/docstrings).
mathjax3_config = {
    "tex": {
        "macros": {
            "abs": [r"\left\lvert #1 \right\rvert", 1],
            "Bignorm": [r"\left\lVert #1 \right\rVert", 1],
            "norm": [r"\lVert #1 \rVert", 1],
            "Biginner": [r"\left\langle #1, #2 \right\rangle", 2],
            "calH": r"\mathcal{H}",
            "IndexFunc": r"\mathcal{I}_{\textup{func}}",
            "IndexOp": r"\mathcal{I}_{\textup{op}}",
            "NumEval": r"\bar{m}",
            "NumEvalFunc": r"\bar{m}_{\textup{func}}",
            "NumEvalOp": r"\bar{m}_{\textup{op}}",
            "NumFunc": r"m_{\textup{func}}",
            "NumOp": r"m_{\textup{op}}",
            "reals": r"\mathbb{R}",
            "prox": r"{\rm{prox}}",
            "Prox": r"\operatorname{Prox}",
            "Id": r"\mathrm{Id}",
            "kron": r"\otimes",
            "inner": [r"\langle #1, #2 \rangle", 2],
            "minimize": r"\operatorname*{minimize}",
            "maximize": r"\operatorname*{maximize}",
            "argmin": r"\operatorname*{argmin}",
            "argmax": r"\operatorname*{argmax}",
            "Argmin": r"\operatorname*{Argmin}",
            "gra": r"\operatorname*{gra}",
            "ran": r"\operatorname*{ran}",
            "zer": r"\operatorname*{zer}",
            "dom": r"\operatorname*{dom}",
            "Fix": r"\operatorname{fix}",
            "sym": r"\mathbb{S}",
            "bx": r"\mathbf{x}",
            "bu": r"\mathbf{u}",
            "by": r"\mathbf{y}",
            "bfcn": r"\mathbf{f}",
            "bFcn": r"\mathbf{F}",
            "bm": [r"\boldsymbol{#1}", 1],
            "llbracket": r"\left[\!\left[",
            "rrbracket": r"\right]\!\right]",
            "naturals": r"\mathbb{N}_{0}",
        }
    }
}


def _suppress_member_toc_entries(app, doctree):
    """Keep class entries in the TOC while hiding member entries."""
    member_objtypes = {
        "method",
        "classmethod",
        "staticmethod",
        "attribute",
        "property",
        "data",
    }
    for node in doctree.findall(addnodes.desc):
        if node.get("domain") != "py":
            continue
        if node.get("objtype") in member_objtypes:
            node["no-contents-entry"] = True
            for sig in node.findall(addnodes.desc_signature):
                sig["no-contents-entry"] = True


def _patch_python_toc_entries():
    """Hide Python member entries from the TOC while keeping class entries."""
    member_objtypes = {
        "method",
        "classmethod",
        "staticmethod",
        "attribute",
        "property",
        "data",
    }
    original = PyObject._toc_entry_name

    def _toc_entry_name(self, sig_node):  # type: ignore[override]
        objtype = sig_node.parent.get("objtype")
        if objtype in member_objtypes:
            return ""
        return original(self, sig_node)

    PyObject._toc_entry_name = _toc_entry_name


def _normalized_baseurl(app):
    """Resolve a canonical base URL from Sphinx config/context."""
    baseurl = (app.config.html_baseurl or "").strip().rstrip("/")
    if baseurl:
        return baseurl

    context_baseurl = app.config.html_context.get("seo_baseurl", "")
    return str(context_baseurl).strip().rstrip("/")


def _inject_seo_page_context(app, pagename, templatename, context, doctree):
    """Expose normalized URL and indexability flags to templates."""
    if app.builder.format != "html":
        return

    baseurl = _normalized_baseurl(app)
    if not baseurl:
        return

    root_doc = app.config.root_doc
    if pagename == root_doc:
        page_url = f"{baseurl}/"
    else:
        page_url = f"{baseurl}/{pagename}.html"

    noindex_pages = {"search", "genindex", "py-modindex", "modindex"}
    is_noindex = (
        pagename in noindex_pages
        or "genindex" in pagename
        or pagename.startswith("_modules/")
        or pagename.startswith("_sources/")
    )

    context["pageurl"] = page_url
    context["seo_page_url"] = page_url
    context["seo_is_noindex"] = is_noindex


def _write_sitemap_and_robots(app, exception):
    """Emit sitemap.xml and robots.txt for HTML builds."""
    if exception is not None or app.builder.format != "html":
        return

    baseurl = _normalized_baseurl(app)
    if not baseurl:
        return

    env = app.builder.env
    outdir = Path(app.builder.outdir)
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]

    for docname in sorted(env.found_docs):
        if docname == app.config.root_doc:
            loc = f"{baseurl}/"
        else:
            loc = f"{baseurl}/{docname}.html"

        source_path = Path(env.doc2path(docname, base=None))
        try:
            lastmod = date.fromtimestamp(source_path.stat().st_mtime).isoformat()
        except OSError:
            lastmod = date.today().isoformat()

        sitemap_lines.extend(
            [
                "  <url>",
                f"    <loc>{xml_escape(loc)}</loc>",
                f"    <lastmod>{lastmod}</lastmod>",
                "  </url>",
            ]
        )

    sitemap_lines.append("</urlset>")
    (outdir / "sitemap.xml").write_text("\n".join(sitemap_lines) + "\n", encoding="utf-8")

    robots_lines = [
        "User-agent: *",
        "Allow: /",
        "Disallow: /search.html",
        "Disallow: /genindex.html",
        "Disallow: /py-modindex.html",
        "Disallow: /_sources/",
        "Disallow: /_modules/",
        f"Sitemap: {baseurl}/sitemap.xml",
    ]
    (outdir / "robots.txt").write_text("\n".join(robots_lines) + "\n", encoding="utf-8")


def setup(app):
    _patch_python_toc_entries()
    app.connect("html-page-context", _inject_seo_page_context)
    app.connect("doctree-read", _suppress_member_toc_entries)
    app.connect("build-finished", _write_sitemap_and_robots)
    # Defer non-critical scripts to reduce render-blocking time.
    app.add_js_file("copybutton.js", defer="defer")
    app.add_js_file("perf.js", defer="defer")

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
