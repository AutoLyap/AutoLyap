import os
import sys
from datetime import date
from pathlib import Path

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

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["copybutton.js"]
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


def setup(app):
    _patch_python_toc_entries()
    app.connect("doctree-read", _suppress_member_toc_entries)

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
