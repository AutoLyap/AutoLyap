import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from docutils import nodes
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

myst_enable_extensions = [
    "amsmath",
]

autodoc_mock_imports = [
    "cvxpy",
    "mosek",
    "mosek.fusion",
    "mosek.fusion.pythonic",
]

autodoc_type_aliases = {
    "CacheValueT": "typing.Any",
    "_IterationIndependentResult": "typing.Dict[str, typing.Any]",
    "_IterationDependentResult": "typing.Dict[str, typing.Any]",
}

templates_path = ["_templates"]
exclude_patterns = ["release_notes/_template.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["custom.css"]
html_baseurl = "https://autolyap.github.io/"
numfig = True
math_numfig = True
numfig_secnum_depth = 1
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
        "theory": {
            "description": (
                "Mathematical background for AutoLyap, including Lyapunov "
                "certificate modeling and SDP-based verification."
            ),
            "keywords": [
                "AutoLyap theory",
                "Lyapunov certificate",
                "semidefinite programming convergence analysis",
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
    "seo_organization_name": project,
    "seo_organization_url": "https://autolyap.github.io",
    "seo_social_profiles": [
        "https://github.com/AutoLyap/AutoLyap",
        "https://manuupadhyaya.github.io/",
    ],
    "seo_og_image_path": "/_static/favicon-master.png",
    "seo_og_image_width": 512,
    "seo_og_image_height": 512,
}
maximum_signature_line_length = 1
toc_object_entries = True
html_use_opensearch = "https://autolyap.github.io"
# Pin MathJax for stable glyph rendering across environments.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"

# MathJax macros aligned with Paper/ver_5/commands.tex and Paper/ver_5/preamble.tex.
mathjax3_config = {
    "loader": {
        "load": ["[tex]/html"],
    },
    "tex": {
        "packages": {"[+]": ["html"]},
        "macros": {
            "abs": [r"\left\lvert #1 \right\rvert", 1],
            "Bignorm": [r"\left\lVert #1 \right\rVert", 1],
            "norm": [r"\lVert #1 \rVert", 1],
            "Biginner": [r"\left\langle #1, #2 \right\rangle", 2],
            "inner": [r"\langle #1, #2 \rangle", 2],
            "reals": r"\mathbb{R}",
            "Rbar": r"\overline{\mathbf{R}}",
            "N": r"\mathbf{N}",
            "naturals": r"\mathbb{N}_{0}",
            "K": r"\mathbf{K}",
            "Or": r"\mathbf{O}",
            "D": r"\mathbf{D}",
            "Sym": r"\mathbf{S}",
            "sym": r"\mathbb{S}",
            "calA": r"\mathcal{A}",
            "calL": r"\mathcal{L}",
            "calH": r"\mathcal{H}",
            "calG": r"\mathcal{G}",
            "calD": r"\mathcal{D}",
            "calT": r"\mathcal{T}",
            "tr": [r"\operatorname{tr}\left(#1\right)", 1],
            "trace": r"\mathrm{trace}",
            "Fix": r"\operatorname{fix}",
            "epi": r"\operatorname{epi}",
            "diag": r"\operatorname{diag}",
            "Range": r"\operatorname{Range}",
            "rank": r"\operatorname{rank}",
            "sgn": r"\operatorname{sgn}",
            "Prox": r"\operatorname{Prox}",
            "prox": r"{\rm{prox}}",
            "kron": r"\otimes",
            "minimize": r"\operatorname*{minimize}",
            "maximize": r"\operatorname*{maximize}",
            "argmax": r"\operatorname*{argmax}",
            "argmin": r"\operatorname*{argmin}",
            "Argmin": r"\operatorname*{Argmin}",
            "adj": r"\operatorname*{adj}",
            "gra": r"\operatorname*{gra}",
            "ran": r"\operatorname*{ran}",
            "zer": r"\operatorname*{zer}",
            "dom": r"\operatorname*{dom}",
            "Id": r"\operatorname*{Id}",
            "Ker": r"\operatorname*{Ker}",
            "Ima": r"\operatorname*{Im}",
            "Cl": r"\operatorname*{cl}",
            "Int": r"\operatorname*{int}",
            "Conv": r"\operatorname*{conv}",
            "quadform": [r"\mathcal{Q}\p{#1,#2}", 2],
            "XId": [r"#1_{\Id}", 1],
            "xmiddle": [r"\;\middle#1\;", 1],
            "allowbreak": r"",
            "bx": r"\mathbf{x}",
            "bu": r"\mathbf{u}",
            "by": r"\mathbf{y}",
            "bz": r"\mathbf{z}",
            "bfcn": r"\mathbf{f}",
            "bFcn": r"\mathbf{F}",
            "bM": r"\mathbf{M}",
            "bMlij": r"\bM_{(l,i,j)}",
            "ba": r"\mathbf{a}",
            "balij": r"\mathbf{a}_{(l,i,j)}",
            "bzeta": r"\boldsymbol{\zeta}",
            "bchi": r"\boldsymbol{\chi}",
            "bxi": r"\boldsymbol{\xi}",
            "bXi": r"\boldsymbol{\Xi}",
            "bQ": r"\mathbf{Q}",
            "bq": r"\mathbf{q}",
            "id": r"I",
            "gramFunc": r"\mathtt{G}",
            "SumToZeroMat": r"N",
            "indentconstr": r"\;\;\;",
            "PEPObjMat": r"W",
            "PEPObjVec": r"w",
            "munderbar": [r"\underline{#1}", 1],
            "PEPMaxIter": r"\bar{k}",
            "PEPMinIter": r"\underline{k}",
            "IndexOp": r"\mathcal{I}_{\textup{op}}",
            "IndexFunc": r"\mathcal{I}_{\textup{func}}",
            "NumFunc": r"m_{\textup{func}}",
            "NumOp": r"m_{\textup{op}}",
            "NumEval": r"\bar{m}",
            "NumEvalOp": r"\bar{m}_{\textup{op}}",
            "NumEvalFunc": r"\bar{m}_{\textup{func}}",
            "set": [r"\mathord{\left.\{ #1 \} \right. }", 1],
            "Bigset": [r"\mathord{\left\{ #1 \right\}}", 1],
            "p": [r"\mathord{( #1 )}", 1],
            "Bigp": [r"\mathord{\left( #1 \right)}", 1],
            "bm": [r"\boldsymbol{#1}", 1],
            "llbracket": r"\lbrack\!\lbrack",
            "rrbracket": r"\rbrack\!\rbrack",
            "underbracket": [r"\underbrace{#1}", 1],
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


def _docname_url_path(app, docname):
    """Resolve a URL path for a docname in the current HTML builder."""
    if docname == app.config.root_doc:
        return "/"

    builder_name = str(getattr(app.builder, "name", "")).strip().lower()
    if builder_name == "dirhtml":
        return f"/{docname}/"

    try:
        target_uri = str(app.builder.get_target_uri(docname)).strip()
    except Exception:
        target_uri = f"{docname}.html"

    if not target_uri:
        return "/"
    return f"/{target_uri.lstrip('/')}"


def _docname_page_url(app, docname, baseurl):
    """Resolve an absolute canonical URL for a docname."""
    return f"{baseurl}{_docname_url_path(app, docname)}"


def _is_noindex_docname(docname: str) -> bool:
    """Return whether a generated HTML page should be excluded from indexing."""
    noindex_pages = {"search", "genindex", "py-modindex", "modindex"}
    return (
        docname in noindex_pages
        or "genindex" in docname
        or docname.startswith("_modules/")
        or docname.startswith("_sources/")
    )


def _normalize_meta_text(text):
    """Collapse whitespace in free-form text for meta tag usage."""
    return " ".join(str(text).split())


def _truncate_meta_description(text, *, max_length=160):
    """Trim text to a sensible meta-description length without mid-word cuts."""
    normalized = _normalize_meta_text(text)
    if len(normalized) <= max_length:
        return normalized

    ellipsis = "..."
    if max_length <= len(ellipsis):
        return ellipsis[:max_length]

    hard_limit = max_length - len(ellipsis)
    cutoff = normalized.rfind(" ", 0, hard_limit + 1)
    if cutoff < int(hard_limit * 0.6):
        cutoff = hard_limit

    truncated = normalized[:cutoff].rstrip(" ,.;:-")
    if not truncated:
        truncated = normalized[:hard_limit]
    return f"{truncated}{ellipsis}"


def _extract_auto_page_description(doctree):
    """Extract a concise page description from the first substantial paragraph."""
    if doctree is None:
        return ""

    for paragraph in doctree.findall(nodes.paragraph):
        candidate = _truncate_meta_description(paragraph.astext())
        if len(candidate) >= 40:
            return candidate
    return ""


def _collect_page_feature_flags(doctree):
    """Return per-page feature flags used to trim optional runtime scripts."""
    flags = {
        "page_has_math": False,
        "page_has_code_blocks": False,
        "page_has_images": False,
    }
    if doctree is None:
        return flags

    flags["page_has_math"] = bool(
        any(doctree.findall(nodes.math)) or any(doctree.findall(nodes.math_block))
    )
    flags["page_has_code_blocks"] = bool(any(doctree.findall(nodes.literal_block)))
    flags["page_has_images"] = bool(any(doctree.findall(nodes.image)))
    return flags


def _script_filename(script_file):
    """Normalize a script file object/string into a comparable filename."""
    filename = getattr(script_file, "filename", "")
    if filename:
        return str(filename)
    return str(script_file)


def _filter_optional_script_files(
    context, *, page_has_math, page_has_code_blocks, page_has_images
):
    """Drop optional scripts from pages that do not need them."""
    script_files = context.get("script_files")
    if not script_files:
        return

    keep_copybutton = bool(page_has_code_blocks)
    keep_math_tag_links = bool(page_has_math)
    keep_perf = bool(page_has_images)
    filtered = []
    for script_file in script_files:
        script_name = _script_filename(script_file)
        if "perf.js" in script_name and not keep_perf:
            continue
        if "copybutton.js" in script_name and not keep_copybutton:
            continue
        if "math_tag_links.js" in script_name and not keep_math_tag_links:
            continue
        filtered.append(script_file)

    context["script_files"] = filtered


def _inject_seo_page_context(app, pagename, templatename, context, doctree):
    """Expose normalized URL and indexability flags to templates."""
    if app.builder.format != "html":
        return

    seo_pages = app.config.html_context.get("seo_pages", {})
    seo_page = seo_pages.get(pagename, {}) if isinstance(seo_pages, dict) else {}
    if not isinstance(seo_page, dict):
        seo_page = {}
    description = seo_page.get("description")
    if description:
        context["seo_page_description"] = _truncate_meta_description(description)
    else:
        context["seo_page_description"] = _extract_auto_page_description(doctree)

    feature_flags = _collect_page_feature_flags(doctree)
    context.update(feature_flags)
    _filter_optional_script_files(
        context,
        page_has_math=feature_flags["page_has_math"],
        page_has_code_blocks=feature_flags["page_has_code_blocks"],
        page_has_images=feature_flags["page_has_images"],
    )

    context["seo_is_noindex"] = _is_noindex_docname(pagename)
    try:
        source_path = Path(app.env.doc2path(pagename, base=None))
        modified_at = datetime.fromtimestamp(
            source_path.stat().st_mtime,
            tz=timezone.utc,
        )
    except Exception:
        modified_at = datetime.now(tz=timezone.utc)

    context["seo_lastmod_iso"] = (
        modified_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    baseurl = _normalized_baseurl(app)
    if not baseurl:
        return

    page_url = _docname_page_url(app, pagename, baseurl)
    context["seo_search_url"] = _docname_page_url(app, "search", baseurl)
    context["pageurl"] = page_url
    context["seo_page_url"] = page_url


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
        if _is_noindex_docname(docname):
            continue
        loc = _docname_page_url(app, docname, baseurl)
        if docname == app.config.root_doc:
            priority = "1.0"
        else:
            priority = "0.8"

        source_path = Path(env.doc2path(docname, base=None))
        try:
            lastmod = (
                datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except OSError:
            lastmod = (
                datetime.now(tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        sitemap_lines.extend(
            [
                "  <url>",
                f"    <loc>{xml_escape(loc)}</loc>",
                f"    <lastmod>{lastmod}</lastmod>",
                "    <changefreq>weekly</changefreq>",
                f"    <priority>{priority}</priority>",
                "  </url>",
            ]
        )

    sitemap_lines.append("</urlset>")
    (outdir / "sitemap.xml").write_text("\n".join(sitemap_lines) + "\n", encoding="utf-8")

    robots_lines = [
        "User-agent: *",
        "Allow: /",
        f"Disallow: {_docname_url_path(app, 'search')}",
        f"Disallow: {_docname_url_path(app, 'genindex')}",
        f"Disallow: {_docname_url_path(app, 'py-modindex')}",
        f"Disallow: {_docname_url_path(app, 'modindex')}",
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
    app.add_js_file("proof_toggle.js", defer="defer")
    app.add_js_file("math_tag_links.js", defer="defer")

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
suppress_warnings = ["bibtex.duplicate_citation"]
