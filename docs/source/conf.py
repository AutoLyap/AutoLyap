# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from xml.sax.saxutils import escape as xml_escape

import tinycss2
from docutils import nodes
from sphinx import addnodes
from sphinx.domains.python._object import PyObject

project = "AutoLyap"
author = "AutoLyap contributors"
copyright = f"{date.today().year}, AutoLyap contributors"

root = Path(__file__).resolve().parents[2]
release = (root / "VERSION").read_text(encoding="utf-8").strip()
version = release

seo_baseurl = "https://autolyap.github.io"
seo_repo_url = "https://github.com/AutoLyap/AutoLyap"
seo_pypi_url = "https://pypi.org/project/autolyap/"
seo_license_url = "https://spdx.org/licenses/GPL-3.0-only.html"
seo_paper_url = "https://doi.org/10.48550/arXiv.2506.24076"

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
html_css_files = ["custom.css"]
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_baseurl = f"{seo_baseurl}/"
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
            "title": "AutoLyap: Automated Lyapunov Analysis for Optimization",
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
            "title": "AutoLyap Quick Start: Certify Optimization Convergence",
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
            "title": "Lyapunov Analysis Theory for First-Order Methods",
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
        "theory/notation_and_preliminaries": {
            "description": (
                "Notation and preliminaries for AutoLyap theory, including "
                "set-valued operator conventions and subdifferential notation."
            ),
            "keywords": [
                "AutoLyap notation and preliminaries",
                "set-valued operator notation",
                "subdifferential conventions",
            ],
        },
        "theory/problem_class": {
            "description": (
                "Theory for problem-class modeling in AutoLyap, including "
                "optimization and inclusion formulations."
            ),
            "keywords": [
                "AutoLyap theory problem class",
                "optimization and inclusion modeling",
                "interpolation indices theory",
            ],
        },
        "theory/algorithm_representation": {
            "description": (
                "Algorithm-representation theory in AutoLyap, including lifted "
                "states and linear recursion modeling."
            ),
            "keywords": [
                "AutoLyap algorithm representation",
                "lifted state recursion",
                "first-order method modeling",
            ],
        },
        "theory/interpolation_conditions": {
            "description": (
                "Interpolation-condition theory in AutoLyap for encoding "
                "function and operator classes as SDP constraints."
            ),
            "keywords": [
                "AutoLyap interpolation conditions",
                "function and operator interpolation",
                "SDP interpolation constraints",
            ],
        },
        "theory/performance_estimation_via_sdps": {
            "description": (
                "Performance-estimation theory via semidefinite programs in "
                "AutoLyap, including Gram-matrix formulations."
            ),
            "keywords": [
                "AutoLyap performance estimation",
                "Gram matrix SDP formulation",
                "worst-case analysis SDP",
            ],
        },
        "theory/lyapunov_analyses": {
            "description": (
                "Lyapunov-analysis theory in AutoLyap, including certificate "
                "inequalities and feasibility interpretations."
            ),
            "keywords": [
                "AutoLyap Lyapunov theory",
                "Lyapunov certificate inequalities",
                "convergence certificate feasibility",
            ],
        },
        "theory/iteration_independent_analyses": {
            "description": (
                "Iteration-independent theory in AutoLyap for certifying "
                "asymptotic linear and sublinear convergence rates."
            ),
            "keywords": [
                "AutoLyap iteration-independent theory",
                "asymptotic convergence certification",
                "linear and sublinear rates",
            ],
        },
        "theory/iteration_dependent_analyses": {
            "description": (
                "Iteration-dependent theory in AutoLyap for finite-horizon "
                "guarantees and chained Lyapunov inequalities."
            ),
            "keywords": [
                "AutoLyap iteration-dependent theory",
                "finite-horizon guarantees",
                "chained Lyapunov inequalities",
            ],
        },
        "api_reference": {
            "title": "AutoLyap Python API Reference for Optimization Analysis",
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
            "title": "First-Order Optimization Algorithms in AutoLyap",
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
            "title": "Optimization Convergence Examples with AutoLyap",
            "description": (
                "Worked AutoLyap examples for gradient, proximal, splitting, "
                "and momentum methods with computer-assisted convergence rates."
            ),
            "keywords": [
                "AutoLyap examples",
                "proximal point method analysis",
                "proximal gradient Lyapunov analysis",
                "heavy-ball method convergence",
                "constant Nesterov momentum",
            ],
        },
        "examples/accelerated_proximal_point": {
            "description": (
                "Accelerated proximal-point example in AutoLyap with "
                "iteration-dependent finite-horizon c_K certification."
            ),
            "keywords": [
                "accelerated proximal point AutoLyap",
                "finite-horizon c_K certificate",
                "iteration-dependent Lyapunov example",
            ],
        },
        "examples/davis_yin_three_operator": {
            "description": (
                "Davis–Yin three-operator splitting example in AutoLyap with "
                "computer-assisted linear-rate rho certification."
            ),
            "keywords": [
                "Davis Yin AutoLyap example",
                "three-operator splitting convergence",
                "rho certification example",
            ],
        },
        "examples/information_theoretic_exact_method": {
            "description": (
                "Information-theoretic exact method example in AutoLyap with "
                "iteration-dependent function-value c_K guarantees."
            ),
            "keywords": [
                "information theoretic exact method AutoLyap",
                "ITEM convergence analysis",
                "function-value c_K certificate",
            ],
        },
        "examples/malitsky_tam_frb": {
            "title": "Malitsky–Tam FRB Convergence Rate",
            "description": (
                "Malitsky–Tam forward-reflected-backward example in AutoLyap "
                "with computer-assisted linear-rate certification."
            ),
            "keywords": [
                "Malitsky Tam FRB AutoLyap",
                "forward reflected backward analysis",
                "linear-rate rho certificate",
            ],
        },
        "examples/nesterov_fast_gradient": {
            "description": (
                "Nesterov fast-gradient example in AutoLyap with "
                "iteration-dependent finite-horizon function-value bounds."
            ),
            "keywords": [
                "Nesterov fast gradient AutoLyap",
                "finite-horizon function-value bound",
                "iteration-dependent c_K analysis",
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
        "examples/triple_momentum": {
            "title": "Triple-Momentum Method: Convergence Rate with AutoLyap",
            "description": (
                "Certify the triple-momentum method's linear rate for smooth "
                "strongly convex optimization using AutoLyap Lyapunov analysis "
                "and the MOSEK Fusion backend."
            ),
            "keywords": [
                "triple momentum method",
                "triple momentum AutoLyap",
                "autolyap.algorithms.TripleMomentum",
                "TMM convergence rate",
                "smooth strongly convex optimization",
                "Lyapunov convergence certificate",
                "linear convergence factor rho",
                "iteration-independent analysis",
                "MOSEK Fusion",
                "Van Scoy Freeman Lynch",
            ],
            "schema_type": ["TechArticle", "LearningResource"],
            "learning_resource_type": "worked example",
            "proficiency_level": "Expert",
            "educational_level": "Advanced",
            "dependencies": "Python, AutoLyap, and optional MOSEK Fusion",
            "teaches": [
                "Model the triple-momentum method in AutoLyap",
                "Search for a certified linear convergence factor rho",
                "Compare the theoretical rate with MOSEK certificates",
            ],
            "citation": {
                "@type": "ScholarlyArticle",
                "name": (
                    "The fastest known globally convergent first-order method "
                    "for minimizing strongly convex functions"
                ),
                "identifier": "https://doi.org/10.1109/LCSYS.2017.2722406",
                "url": "https://doi.org/10.1109/LCSYS.2017.2722406",
                "author": [
                    {"@type": "Person", "name": "Bryan Van Scoy"},
                    {"@type": "Person", "name": "Randy A. Freeman"},
                    {"@type": "Person", "name": "Kevin M. Lynch"},
                ],
            },
        },
        "examples/gradient_method/index": {
            "description": (
                "Gradient-method examples in AutoLyap for Lyapunov-based "
                "convergence analysis under different problem settings."
            ),
            "keywords": [
                "gradient method examples",
                "AutoLyap gradient method",
                "Lyapunov convergence certificates",
            ],
        },
        "examples/gradient_method/smooth_strongly_convex": {
            "description": (
                "Gradient-method smooth strongly-convex example in AutoLyap with "
                "iteration-independent Lyapunov analysis and bisection search."
            ),
            "keywords": [
                "gradient method smooth strongly convex",
                "AutoLyap bisection rho",
                "distance-to-solution convergence",
            ],
        },
        "examples/gradient_method/gradient_dominated_smooth": {
            "title": "Gradient Method: Smooth Gradient-Dominated Rate",
            "description": (
                "Gradient-method gradient-dominated smooth example in AutoLyap "
                "with certified linear function-value rates."
            ),
            "keywords": [
                "gradient method nonconvex",
                "gradient dominated smooth",
                "AutoLyap function-value convergence",
            ],
        },
        "examples/optimized_gradient": {
            "description": (
                "Optimized gradient method example in AutoLyap with "
                "iteration-dependent Lyapunov analysis and finite-horizon "
                "function-value guarantees."
            ),
            "keywords": [
                "optimized gradient method AutoLyap",
                "iteration-dependent Lyapunov analysis",
                "finite-horizon c_K certificate",
            ],
        },
        "examples/chambolle_pock/index": {
            "description": (
                "Chambolle–Pock examples in AutoLyap for Lyapunov-based "
                "analysis under multiple problem settings."
            ),
            "keywords": [
                "Chambolle Pock AutoLyap",
                "Chambolle Pock Lyapunov analysis",
                "fixed-point residual and linear-rate examples",
            ],
        },
        "examples/chambolle_pock/convex_fixed_point_residual": {
            "description": (
                "Chambolle–Pock convex example in AutoLyap with "
                "fixed-point-residual summability certification and layered "
                "(h, alpha) regions."
            ),
            "keywords": [
                "Chambolle Pock convex fixed-point residual",
                "fixed-point residual summability",
                "history overlap Lyapunov analysis",
            ],
        },
        "examples/chambolle_pock/smooth_strongly_convex": {
            "title": "Chambolle–Pock: Smooth Strongly Convex Rate",
            "description": (
                "Chambolle–Pock smooth strongly-convex example in AutoLyap "
                "with iteration-independent linear-rate certification."
            ),
            "keywords": [
                "Chambolle Pock smooth strongly convex",
                "AutoLyap bisection rho",
                "distance-to-solution convergence",
            ],
        },
        "examples/define_your_own_algorithm/index": {
            "description": (
                "Examples in AutoLyap for defining custom algorithms from the "
                "base Algorithm interface."
            ),
            "keywords": [
                "define your own algorithm",
                "AutoLyap custom algorithm examples",
                "Algorithm interface examples",
            ],
        },
        "examples/define_your_own_algorithm/proximal_gradient_method": {
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
        "examples/douglas_rachford/index": {
            "description": (
                "Douglas–Rachford examples in AutoLyap covering cocoercive, "
                "Lipschitz, and smooth strongly-convex problem settings."
            ),
            "keywords": [
                "Douglas Rachford AutoLyap examples",
                "operator splitting Lyapunov analysis",
                "rho certification across settings",
            ],
        },
        "examples/douglas_rachford/cocoercive_plus_strongly_monotone": {
            "title": "Douglas–Rachford: Cocoercive + Strongly Monotone",
            "description": (
                "Douglas–Rachford cocoercive-plus-strongly-monotone example "
                "in AutoLyap with linear-rate rho certification."
            ),
            "keywords": [
                "Douglas Rachford cocoercive strongly monotone",
                "AutoLyap operator splitting example",
                "rho versus lambda analysis",
            ],
        },
        "examples/douglas_rachford/maximally_monotone_lipschitz_plus_strongly_monotone": {
            "title": "Douglas–Rachford: Lipschitz + Strongly Monotone",
            "description": (
                "Douglas–Rachford example for maximally-monotone-Lipschitz "
                "plus strongly-monotone operators with certified rates."
            ),
            "keywords": [
                "Douglas Rachford maximally monotone Lipschitz",
                "strongly monotone operator splitting",
                "AutoLyap linear-rate certificate",
            ],
        },
        "examples/douglas_rachford/maximally_monotone_plus_strongly_monotone_cocoercive": {
            "title": "Douglas–Rachford: Strongly Monotone/Cocoercive",
            "description": (
                "Douglas–Rachford example for maximally-monotone plus "
                "strongly-monotone-cocoercive operators with rho certification."
            ),
            "keywords": [
                "Douglas Rachford cocoercive operator example",
                "maximally monotone plus strongly monotone",
                "AutoLyap rho certification",
            ],
        },
        "examples/douglas_rachford/maximally_monotone_plus_strongly_monotone_lipschitz": {
            "title": "Douglas–Rachford: Strongly Monotone/Lipschitz",
            "description": (
                "Douglas–Rachford example for maximally-monotone plus "
                "strongly-monotone-Lipschitz operators with certified rates."
            ),
            "keywords": [
                "Douglas Rachford strongly monotone Lipschitz",
                "operator splitting SDP certificate",
                "AutoLyap rho versus gamma",
            ],
        },
        "examples/douglas_rachford/smooth_strongly_convex_plus_convex": {
            "title": "Douglas–Rachford: Smooth Strongly Convex + Convex",
            "description": (
                "Douglas–Rachford smooth-strongly-convex-plus-convex example "
                "in AutoLyap with iteration-independent linear-rate analysis."
            ),
            "keywords": [
                "Douglas Rachford smooth strongly convex plus convex",
                "AutoLyap iteration-independent example",
                "linear-rate rho certification",
            ],
        },
        "examples/heavy_ball/index": {
            "description": (
                "Heavy-ball examples in AutoLyap for Lyapunov-based "
                "convergence analysis under different problem settings."
            ),
            "keywords": [
                "heavy-ball method analysis",
                "AutoLyap heavy-ball examples",
                "Lyapunov convergence certificates",
            ],
        },
        "examples/heavy_ball/smooth_convex": {
            "description": (
                "Heavy-ball smooth-convex example in AutoLyap with certified "
                "sublinear function-value convergence."
            ),
            "keywords": [
                "heavy-ball smooth convex",
                "AutoLyap heavy-ball",
                "sublinear convergence certificate",
            ],
        },
        "examples/heavy_ball/gradient_dominated_smooth": {
            "title": "Heavy-Ball: Smooth Gradient-Dominated Rate",
            "description": (
                "Heavy-ball gradient-dominated-smooth example in AutoLyap with "
                "computer-assisted linear function-value rate certification."
            ),
            "keywords": [
                "heavy-ball gradient-dominated smooth",
                "AutoLyap heavy-ball nonconvex example",
                "linear function-value convergence",
            ],
        },
        "examples/nesterov_momentum/index": {
            "title": "Constant Nesterov Momentum Examples",
            "description": (
                "Constant Nesterov momentum examples in AutoLyap for "
                "Lyapunov-based convergence analysis under different "
                "problem settings."
            ),
            "keywords": [
                "constant Nesterov momentum",
                "AutoLyap momentum examples",
                "Lyapunov convergence certificates",
            ],
        },
        "examples/nesterov_momentum/smooth_convex": {
            "title": "Constant Nesterov Momentum: Smooth Convex",
            "description": (
                "Constant Nesterov momentum smooth-convex example in AutoLyap "
                "with certified sublinear function-value convergence."
            ),
            "keywords": [
                "constant Nesterov momentum smooth convex",
                "AutoLyap Nesterov momentum",
                "sublinear convergence certificate",
            ],
        },
        "examples/nesterov_momentum/gradient_dominated_smooth": {
            "title": "Constant Nesterov Momentum: Gradient-Dominated",
            "description": (
                "Constant Nesterov momentum gradient-dominated smooth example "
                "in AutoLyap with certified linear rates."
            ),
            "keywords": [
                "constant Nesterov momentum nonconvex",
                "gradient dominated smooth",
                "AutoLyap bisection rho",
            ],
        },
        "function_classes": {
            "title": "Convex and Smooth Function Classes | AutoLyap",
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
            "title": "Monotone Operator Classes | AutoLyap",
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
            "title": "Optimization Problem Classes in AutoLyap",
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
            "title": "Iteration-Independent Lyapunov Analysis | AutoLyap",
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
            "title": "Iteration-Dependent Lyapunov Analysis | AutoLyap",
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
            "title": "Lyapunov Convergence Analysis API | AutoLyap",
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
            "title": "MOSEK Fusion and CVXPY Solver Backends | AutoLyap",
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
        "contributing/getting_started": {
            "title": "Get Started Contributing to AutoLyap",
            "description": (
                "Set up an AutoLyap development environment, install test and "
                "documentation dependencies, and verify a local checkout."
            ),
            "keywords": [
                "AutoLyap contributor setup",
                "AutoLyap development environment",
                "install AutoLyap from source",
            ],
        },
        "contributing/development_workflow": {
            "title": "AutoLyap Development and Testing Workflow",
            "description": (
                "Follow the AutoLyap development workflow for branches, tests, "
                "documentation builds, formatting, and pre-commit validation."
            ),
            "keywords": [
                "AutoLyap development workflow",
                "AutoLyap testing guide",
                "AutoLyap documentation build",
            ],
        },
        "contributing/pull_request_process": {
            "title": "AutoLyap Pull Request Process",
            "description": (
                "Prepare, validate, and submit an AutoLyap pull request with the "
                "project's review checklist and contribution requirements."
            ),
            "keywords": [
                "AutoLyap pull request",
                "AutoLyap contribution checklist",
                "contribute code to AutoLyap",
            ],
        },
        "dev/dev_reference": {
            "title": "AutoLyap Developer API and Internals Reference",
            "description": (
                "Developer reference for AutoLyap internals, including analysis "
                "assembly, solver execution, algorithms, and problem classes."
            ),
            "keywords": [
                "AutoLyap developer reference",
                "AutoLyap internal API",
                "AutoLyap architecture",
            ],
        },
        "dev/dev_internal_algorithms": {
            "title": "AutoLyap Internal Algorithm Modules",
            "description": (
                "Internal AutoLyap algorithm APIs for contributors implementing "
                "or modifying first-order optimization methods."
            ),
            "keywords": [
                "AutoLyap algorithm internals",
                "internal optimization algorithm API",
                "AutoLyap contributor reference",
            ],
        },
        "dev/dev_internal_core": {
            "title": "AutoLyap Internal Analysis and Solver Modules",
            "description": (
                "Internal AutoLyap APIs for Lyapunov analysis assembly, solver "
                "backends, diagnostics, and certificate execution."
            ),
            "keywords": [
                "AutoLyap analysis internals",
                "AutoLyap solver internals",
                "Lyapunov certificate implementation",
            ],
        },
        "dev/dev_internal_problemclass": {
            "title": "AutoLyap Internal Problem-Class Modules",
            "description": (
                "Internal AutoLyap problem-class APIs for function, operator, "
                "inclusion-problem, and interpolation-index implementations."
            ),
            "keywords": [
                "AutoLyap problem class internals",
                "interpolation index implementation",
                "operator class internal API",
            ],
        },
        "dev/dev_internal_utils": {
            "title": "AutoLyap Internal Utility Modules",
            "description": (
                "Internal AutoLyap utility APIs shared by algorithm, problem "
                "class, analysis, validation, and solver modules."
            ),
            "keywords": [
                "AutoLyap utility internals",
                "AutoLyap validation helpers",
                "AutoLyap backend types",
            ],
        },
        "dev/dev_external_reference_targets": {
            "title": "AutoLyap External Reference Targets",
            "description": (
                "Internal cross-reference targets used to resolve external Python "
                "types while building the AutoLyap documentation."
            ),
            "keywords": [
                "AutoLyap external reference targets",
                "Sphinx Python cross references",
                "AutoLyap documentation internals",
            ],
            "noindex": True,
        },
        "examples/scripts/README": {
            "title": "AutoLyap Example Asset Scripts",
            "description": (
                "Developer catalog of scripts that regenerate AutoLyap example "
                "datasets and plots for the documentation."
            ),
            "keywords": [
                "AutoLyap example scripts",
                "AutoLyap documentation assets",
                "regenerate AutoLyap plots",
            ],
            "noindex": True,
        },
        "whats_new": {
            "title": "AutoLyap Release Notes and New Features",
            "description": (
                "Explore AutoLyap release highlights, new analysis features, "
                "documentation updates, and changes across package versions."
            ),
            "keywords": [
                "AutoLyap release notes",
                "AutoLyap changelog",
                "AutoLyap v0.2.1",
            ],
        },
        "release_notes/v0_2_1": {
            "description": (
                "AutoLyap v0.2.1 release notes covering notebooks and "
                "documentation consistency updates."
            ),
            "keywords": [
                "AutoLyap v0.2.1",
                "AutoLyap patch release",
                "AutoLyap release notes",
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
    "seo_repo_url": seo_repo_url,
    "seo_baseurl": seo_baseurl,
    "seo_pypi_url": seo_pypi_url,
    "seo_license_url": seo_license_url,
    "seo_paper_url": seo_paper_url,
    "seo_author": author,
    "seo_contributors": [
        {"@type": "Person", "name": "Manu Upadhyaya"},
        {"@type": "Person", "name": "Shuvomoy Das Gupta"},
        {"@type": "Person", "name": "Adrien B. Taylor"},
        {"@type": "Person", "name": "Sebastian Banert"},
        {"@type": "Person", "name": "Pontus Giselsson"},
    ],
    "seo_in_language": "en-US",
    "seo_organization_name": project,
    "seo_organization_url": seo_baseurl,
    "seo_social_profiles": [
        "https://github.com/AutoLyap",
        seo_pypi_url,
    ],
    "seo_og_image_path": "/_static/favicon-master.png",
    "seo_og_image_width": 512,
    "seo_og_image_height": 512,
}
maximum_signature_line_length = 1
toc_object_entries = True
html_use_opensearch = "https://autolyap.github.io"
# Pin MathJax for stable glyph rendering across environments.  The docs contain
# TeX input only, so omit the unused MathML input component from the bundle.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"
mathjax_options = {"defer": "defer", "crossorigin": "anonymous"}

# MathJax macros aligned with Paper/ver_5/commands.tex and Paper/ver_5/preamble.tex.
mathjax3_config = {
    "loader": {
        # Math-heavy reference pages contain hundreds of expressions.  MathJax's
        # lazy component keeps the initial render bounded to the viewport while
        # preserving the same CommonHTML output as readers scroll.
        "load": ["ui/lazy"],
    },
    "options": {
        "lazyMargin": "200px",
        # Preserve the initial viewport's geometry before lazy observation starts.
        "lazyAlwaysTypeset": [".math.math-initial"],
    },
    "tex": {
        "packages": {"[+]": []},
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
        },
    },
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


def _is_noindex_docname(docname: str, seo_pages=None) -> bool:
    """Return whether a generated HTML page should be excluded from indexing."""
    noindex_pages = {"search", "genindex", "py-modindex", "modindex"}
    generated_noindex = (
        docname in noindex_pages
        or "genindex" in docname
        or docname.startswith("_modules/")
        or docname.startswith("_sources/")
    )
    if generated_noindex:
        return True

    if not isinstance(seo_pages, dict):
        return False
    page_config = seo_pages.get(docname, {})
    return isinstance(page_config, dict) and bool(page_config.get("noindex"))


def _normalize_meta_text(text):
    """Collapse whitespace in free-form text for meta tag usage."""
    normalized = re.sub(r"<[^>]+>", " ", str(text))
    # Reduce LaTeX-heavy inline text (common in theory pages) to plain tokens
    # so meta descriptions and JSON-LD fields remain readable and valid.
    normalized = re.sub(r"\\([A-Za-z]+)", r"\1", normalized)
    normalized = normalized.replace("{", "").replace("}", "").replace("$", "")
    normalized = normalized.replace("\\", " ")
    return " ".join(normalized.split())


def _truncate_meta_description(text, *, max_length=160):
    """Trim text to a sensible meta-description length without mid-word cuts."""
    normalized = _normalize_meta_text(text)
    if len(normalized) <= max_length:
        return normalized.rstrip(" ,;:-")

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


def _is_math_heavy_meta_candidate(text):
    """Return whether text is likely math-dense and poor as an SEO description."""
    raw = str(text)
    if "\\" not in raw and "$" not in raw:
        return False
    return len(re.findall(r"\\[A-Za-z]+", raw)) >= 2


def _has_math_artifact_tokens(text):
    """Detect leftover TeX-like tokens that hurt snippet readability."""
    tokenized = str(text).lower()
    return any(
        marker in tokenized
        for marker in (
            "mathbb",
            "mathcal",
            "operatorname",
            "left",
            "right",
            "infty",
            "subset",
            "supset",
            "cup",
            "cap",
        )
    )


def _build_fallback_description(page_title, project_name):
    """Build a readable fallback description when no paragraph is suitable."""
    title_text = _normalize_meta_text(page_title)
    if not title_text:
        return ""
    if project_name.lower() in title_text.lower():
        return _truncate_meta_description(f"{title_text}.")
    return _truncate_meta_description(f"{title_text} documentation for {project_name}.")


def _build_fallback_keywords(pagename, page_title, default_keywords):
    """Generate page-specific keywords when explicit per-page keywords are absent."""
    keywords = []
    seen = set()

    def _push(keyword):
        if not keyword:
            return
        normalized = _normalize_meta_text(keyword).strip(" ,;:-")
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        keywords.append(normalized)

    title_text = _normalize_meta_text(page_title)
    if title_text and title_text.lower() != "autolyap":
        _push(f"AutoLyap {title_text}")

    parts = [p for p in str(pagename).split("/") if p and p != "index"]
    if parts:
        readable_path = _normalize_meta_text(
            " ".join(re.sub(r"[_-]+", " ", part) for part in parts)
        )
        if readable_path:
            _push(f"AutoLyap {readable_path}")

    for keyword in default_keywords:
        _push(keyword)

    return keywords[:10]


def _extract_auto_page_description(doctree):
    """Extract a concise page description from the first substantial paragraph."""
    if doctree is None:
        return ""

    for paragraph in doctree.findall(nodes.paragraph):
        raw_text = paragraph.astext()
        if _is_math_heavy_meta_candidate(raw_text):
            continue
        candidate = _truncate_meta_description(paragraph.astext())
        if _has_math_artifact_tokens(candidate):
            continue
        if candidate.lower().endswith(("i.e.", "e.g.", "etc.")):
            continue
        if len(candidate) >= 40:
            return candidate
    return ""


def _collect_page_feature_flags(doctree):
    """Return per-page feature flags used to trim optional runtime scripts."""
    flags = {
        "page_has_math": False,
        "page_has_code_blocks": False,
        "page_has_images": False,
        "page_has_proofs": False,
        "page_needs_math_html": False,
        "page_needs_math_tag_links": False,
    }
    if doctree is None:
        return flags

    flags["page_has_math"] = bool(
        any(doctree.findall(nodes.math)) or any(doctree.findall(nodes.math_block))
    )
    flags["page_has_code_blocks"] = bool(any(doctree.findall(nodes.literal_block)))
    flags["page_has_images"] = bool(any(doctree.findall(nodes.image)))
    flags["page_has_proofs"] = bool(
        any(
            isinstance(container, nodes.container)
            and "proof" in container.get("classes", [])
            for container in doctree.findall(nodes.container)
        )
    )
    math_nodes = [
        *doctree.findall(nodes.math),
        *doctree.findall(nodes.math_block),
    ]
    math_source = "\n".join(str(node.rawsource) for node in math_nodes)
    flags["page_needs_math_html"] = bool(
        re.search(r"\\(?:href|class|cssId|style)\b", math_source)
    )
    flags["page_needs_math_tag_links"] = bool(
        r"\tag{" in math_source
        or any(
            str(css_class).startswith("eq-align-")
            for element in doctree.findall()
            if isinstance(element, nodes.Element)
            for css_class in element.get("classes", [])
        )
        or any(
            str(element_id) in {"eq-c1", "eq-c2", "eq-c3", "eq-c4"}
            for element in doctree.findall()
            if isinstance(element, nodes.Element)
            for element_id in element.get("ids", [])
        )
    )
    return flags


def _script_filename(script_file):
    """Normalize a script file object/string into a comparable filename."""
    filename = getattr(script_file, "filename", "")
    if filename:
        return str(filename)
    return str(script_file)


def _filter_optional_script_files(
    context, *, page_has_code_blocks, page_has_proofs, page_needs_math_tag_links
):
    """Drop optional scripts from pages that do not need them."""
    script_files = context.get("script_files")
    if not script_files:
        return

    keep_copybutton = bool(page_has_code_blocks)
    keep_math_tag_links = bool(page_needs_math_tag_links)
    keep_proof_toggle = bool(page_has_proofs)
    filtered = []
    for script_file in script_files:
        script_name = _script_filename(script_file)
        # Sphinx emits this legacy jQuery shim when a theme registers jQuery.
        # Current Sphinx, RTD theme, and authored scripts use none of its APIs.
        if "_sphinx_javascript_frameworks_compat.js" in script_name:
            continue
        if "copybutton.js" in script_name and not keep_copybutton:
            continue
        if "math_tag_links.js" in script_name and not keep_math_tag_links:
            continue
        if "proof_toggle.js" in script_name and not keep_proof_toggle:
            continue
        filtered.append(script_file)

    context["script_files"] = filtered


def _filter_optional_css_files(context, *, page_has_code_blocks):
    """Drop syntax-highlighting CSS from pages without literal blocks."""
    css_files = context.get("css_files")
    if not css_files or page_has_code_blocks:
        return
    context["css_files"] = [
        css_file
        for css_file in css_files
        if "pygments.css" not in _script_filename(css_file)
    ]


_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*?/?>", flags=re.IGNORECASE)
_HTML_CONTAINER_RE = re.compile(r"<(?:div|span)\b[^>]*>", flags=re.IGNORECASE)
_NEUTRAL_PRE_SPAN_RE = re.compile(
    r"<span\s+class=([\"'])pre\1>(?P<text>[^<]*)</span>",
    flags=re.IGNORECASE,
)
_WHITESPACE_PYGMENTS_SPAN_RE = re.compile(
    r"<span\s+class=([\"'])w\1>(?P<text>\s*)</span>",
    flags=re.IGNORECASE,
)
_EMPTY_SPAN_RE = re.compile(r"<span\s*></span>", flags=re.IGNORECASE)
_HTML_SPAN_TAG_RE = re.compile(r"<span\b[^>]*>|</span\s*>", flags=re.IGNORECASE)
_NEUTRAL_PYGMENTS_OPENING_RE = re.compile(
    r"<span\s+class=([\"'])(?:n|p)\1\s*>", flags=re.IGNORECASE
)
_NEUTRAL_PRE_SELECTOR_RE = re.compile(r"\.pre(?![\w-])")
_NEUTRAL_PYGMENTS_SELECTOR_RE = re.compile(r"\.(?:n|p)(?![\w-])")
_BADGE_LINK_RE = re.compile(
    r"<a\b[^>]*>\s*<img\b[^>]*\bsrc=[\"']https://img\.shields\.io/[^>]*?/?>\s*</a>",
    flags=re.IGNORECASE,
)
_SHIELDS_IMAGE_DIMENSIONS = {
    "PyPI version": (97, 20),
    "PyPI downloads": (104, 20),
    "GitHub stars": (77, 20),
    "Paper": (169, 20),
    "Open in Colab": (111, 20),
}
_DOMAIN_GREEK_CHARACTERS = "ΓΔΘΛΞΟΠΣΦΨΩαβγδεζηθικλμνξοπρστυφχψω"
_MATHJAX_CONFIG_RE = re.compile(
    r"(?P<prefix><script>window\.MathJax = )(?P<config>\{.*?\})(?P<suffix></script>)",
    flags=re.DOTALL,
)
_CUSTOM_TEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")


def _html_attribute(tag, name):
    """Return an HTML attribute value from a generated element string."""
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*([\"'])(.*?)\1",
        tag,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(2) if match else ""


def _append_html_attribute(tag, name, value):
    """Add an attribute to a generated HTML tag unless it already exists."""
    if re.search(rf"\b{re.escape(name)}\s*=", tag, flags=re.IGNORECASE):
        return tag
    if tag.endswith("/>"):
        return f'{tag[:-2].rstrip()} {name}="{value}" />'
    return f'{tag[:-1].rstrip()} {name}="{value}">'


def _append_html_class(tag, class_name):
    """Append one class to a generated element without disturbing its markup."""
    class_match = re.search(
        r"\bclass\s*=\s*([\"'])(.*?)\1",
        tag,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if class_match is None:
        return _append_html_attribute(tag, "class", class_name)
    classes = class_match.group(2).split()
    if class_name in classes:
        return tag
    classes.append(class_name)
    start, end = class_match.span(2)
    return f"{tag[:start]}{' '.join(classes)}{tag[end:]}"


def _mark_initial_math(markup, limit=4):
    """Eagerly typeset the math that can influence initial viewport geometry."""
    marked = 0

    def _mark_container(match):
        nonlocal marked
        tag = match.group(0)
        classes = _html_attribute(tag, "class").split()
        if marked >= limit or "math" not in classes:
            return tag
        marked += 1
        return _append_html_class(tag, "math-initial")

    return _HTML_CONTAINER_RE.sub(_mark_container, markup)


def _unwrap_neutral_pygments_spans(markup):
    """Unwrap exact ``n``/``p`` token spans while preserving nested markup."""
    stack = []
    removals = []
    for match in _HTML_SPAN_TAG_RE.finditer(markup):
        tag = match.group(0)
        if tag.lower().startswith("</span"):
            if not stack:
                continue
            opening, should_unwrap = stack.pop()
            if should_unwrap:
                removals.extend((opening.span(), match.span()))
            continue
        if tag.rstrip().endswith("/>"):
            continue
        stack.append((match, bool(_NEUTRAL_PYGMENTS_OPENING_RE.fullmatch(tag))))

    for start, end in sorted(removals, reverse=True):
        markup = f"{markup[:start]}{markup[end:]}"
    return markup


def _macro_definition_source(value):
    """Return the TeX source from one MathJax macro definition value."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return ""


def _specialize_mathjax_config(markup):
    """Keep only page-used macros and skip lazy loading when all math is eager."""
    match = _MATHJAX_CONFIG_RE.search(markup)
    if match is None:
        return markup
    config = json.loads(match.group("config"))
    configured_macros = config.get("tex", {}).get("macros", {})
    page_source = f"{markup[: match.start()]}{markup[match.end() :]}"
    required_macros = set(_CUSTOM_TEX_COMMAND_RE.findall(page_source)) & set(
        configured_macros
    )
    pending = list(required_macros)
    while pending:
        macro_name = pending.pop()
        dependencies = set(
            _CUSTOM_TEX_COMMAND_RE.findall(
                _macro_definition_source(configured_macros[macro_name])
            )
        ) & set(configured_macros)
        for dependency in dependencies - required_macros:
            required_macros.add(dependency)
            pending.append(dependency)
    config["tex"]["macros"] = {
        name: value
        for name, value in configured_macros.items()
        if name in required_macros
    }

    math_containers = [
        _html_attribute(tag, "class").split()
        for tag in _HTML_CONTAINER_RE.findall(page_source)
        if "math" in _html_attribute(tag, "class").split()
    ]
    if math_containers and all(
        "math-initial" in classes for classes in math_containers
    ):
        loader = config.get("loader", {})
        loader["load"] = [
            component for component in loader.get("load", []) if component != "ui/lazy"
        ]
        if not loader.get("load"):
            config.pop("loader", None)
        options = config.get("options", {})
        options.pop("lazyMargin", None)
        options.pop("lazyAlwaysTypeset", None)
        if not options:
            config.pop("options", None)

    serialized = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    replacement = f"{match.group('prefix')}{serialized}{match.group('suffix')}"
    return f"{markup[: match.start()]}{replacement}{markup[match.end() :]}"


@lru_cache(maxsize=None)
def _svg_intrinsic_dimensions(path_string):
    """Read integer intrinsic dimensions from one of the generated plot SVGs."""
    path = Path(path_string)
    if not path.is_file():
        return None
    header = path.read_text(encoding="utf-8")[:2048]
    svg_tag = re.search(r"<svg\b[^>]*>", header, flags=re.IGNORECASE)
    if svg_tag is None:
        return None
    width = _html_attribute(svg_tag.group(0), "width")
    height = _html_attribute(svg_tag.group(0), "height")
    if not width.isdigit() or not height.isdigit():
        return None
    return int(width), int(height)


def _optimize_content_markup(app, context):
    """Emit stable loading hints before the browser discovers page resources."""
    body = context.get("body")
    if not body:
        return

    static_dir = Path(app.srcdir) / "_static"

    def _enhance_image(match):
        tag = match.group(0)
        source = _html_attribute(tag, "src")
        alt = _html_attribute(tag, "alt")
        parsed_source = urlparse(source)
        is_badge = parsed_source.netloc == "img.shields.io"

        dimensions = _SHIELDS_IMAGE_DIMENSIONS.get(alt) if is_badge else None
        if dimensions is None and parsed_source.path.lower().endswith(".svg"):
            source_name = Path(unquote(parsed_source.path)).name
            dimensions = _svg_intrinsic_dimensions(str(static_dir / source_name))
        if dimensions is not None:
            tag = _append_html_attribute(tag, "width", dimensions[0])
            tag = _append_html_attribute(tag, "height", dimensions[1])

        tag = _append_html_attribute(tag, "decoding", "async")
        if is_badge:
            tag = _append_html_attribute(tag, "loading", "eager")
            # Badges are visible immediately, but they must not compete with the
            # document fonts and render-blocking stylesheets on a cold load.
            tag = _append_html_attribute(tag, "fetchpriority", "low")
        else:
            tag = _append_html_attribute(tag, "loading", "lazy")
            tag = _append_html_attribute(tag, "fetchpriority", "low")
        return tag

    optimized_body = _HTML_IMAGE_RE.sub(_enhance_image, str(body))
    optimized_body = _mark_initial_math(optimized_body)

    def _enhance_badge_link(match):
        markup = match.group(0)
        opening_tag_end = markup.find(">") + 1
        opening_tag = markup[:opening_tag_end]
        opening_tag = _append_html_attribute(opening_tag, "target", "_blank")
        opening_tag = _append_html_attribute(opening_tag, "rel", "noopener noreferrer")
        return f"{opening_tag}{markup[opening_tag_end:]}"

    context["body"] = _BADGE_LINK_RE.sub(_enhance_badge_link, optimized_body)


def _font_subset_codepoints(outdir):
    """Collect visible current-site characters for exact, rebuildable subsets."""

    class _VisibleTextParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.characters = set()
            self._hidden_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style"}:
                self._hidden_depth += 1

        def handle_endtag(self, tag):
            if tag in {"script", "style"} and self._hidden_depth:
                self._hidden_depth -= 1

        def handle_data(self, data):
            if not self._hidden_depth:
                self.characters.update(ord(character) for character in data)

    codepoints = set(range(0x20, 0x7F))
    codepoints.add(0x00A0)
    for html_path in Path(outdir).rglob("*.html"):
        parser = _VisibleTextParser()
        parser.feed(html_path.read_text("utf-8"))
        codepoints.update(parser.characters)
    return {codepoint for codepoint in codepoints if not 0xE000 <= codepoint <= 0xF8FF}


def _font_greek_codepoints():
    """Return domain symbols that can appear only in runtime search queries."""
    return set(map(ord, _DOMAIN_GREEK_CHARACTERS))


def _fontawesome_subset_codepoints(outdir):
    """Collect FontAwesome glyphs used by markup and RTD structural controls."""
    codepoints = {
        0xF019,  # download link
        0xF02D,  # Read the Docs version book
        0xF057,  # validation error
        0xF058,  # validation success
        0xF06A,  # admonition/validation notice
        0xF08E,  # external-link marker
        0xF0A8,  # previous page
        0xF0A9,  # next page
        0xF0C1,  # heading permalink
        0xF0C9,  # mobile navigation
        0xF0D7,  # dropdown caret
        0xF147,  # expanded navigation branch
        0xF196,  # collapsed navigation branch
    }
    for html_path in Path(outdir).rglob("*.html"):
        codepoints.update(
            ord(character)
            for character in html_path.read_text("utf-8")
            if 0xF000 <= ord(character) <= 0xF8FF
        )
    return codepoints


def _built_image_dimensions(path):
    """Return intrinsic dimensions for SVG and PNG build assets."""
    if path.suffix.lower() == ".svg":
        return _svg_intrinsic_dimensions(str(path))
    if path.suffix.lower() == ".png" and path.is_file():
        header = path.read_bytes()[:24]
        if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 24:
            return int.from_bytes(header[16:20], "big"), int.from_bytes(
                header[20:24], "big"
            )
    return None


def _finalize_generated_markup(outdir):
    """Cover generated index/search markup that bypasses document doctrees."""
    static_dir = Path(outdir) / "_static"
    html_paths = list(Path(outdir).rglob("*.html"))
    generated_markup = "\n".join(
        html_path.read_text(encoding="utf-8") for html_path in html_paths
    )
    active_assets = [
        asset_path
        for asset_path in [
            *static_dir.rglob("*.css"),
            *static_dir.rglob("*.js"),
        ]
        if asset_path.name in generated_markup
    ]
    styled_pre_assets = [
        asset_path
        for asset_path in active_assets
        if _NEUTRAL_PRE_SELECTOR_RE.search(
            asset_path.read_text(encoding="utf-8", errors="ignore")
        )
    ]
    if styled_pre_assets:
        raise RuntimeError(
            "refusing to unwrap styled Sphinx .pre spans: "
            + ", ".join(str(path) for path in styled_pre_assets)
        )
    pygments_span_assets = [
        asset_path
        for asset_path in active_assets
        if asset_path.suffix == ".css"
        and _NEUTRAL_PYGMENTS_SELECTOR_RE.search(
            asset_path.read_text(encoding="utf-8", errors="ignore")
        )
    ]
    if pygments_span_assets:
        raise RuntimeError(
            "refusing to unwrap active Pygments token spans: "
            + ", ".join(str(path) for path in pygments_span_assets)
        )

    external_script_re = re.compile(
        r"<script\b(?=[^>]*\bsrc=)[^>]*>", flags=re.IGNORECASE
    )
    navigation_bootstrap_re = re.compile(
        r"<script>\s*jQuery\(function \(\) \{\s*"
        r"SphinxRtdTheme\.Navigation\.enable\((true|false)\);\s*"
        r"\}\);\s*</script>",
        flags=re.IGNORECASE,
    )
    signature_parameter_list_re = re.compile(
        r"(<dl>\s*)(?=<dd><em class=[\"']sig-param[\"'])",
        flags=re.IGNORECASE,
    )

    for html_path in html_paths:
        markup = html_path.read_text("utf-8")
        # Sphinx wraps literal text in class-only spans.  No shipped stylesheet
        # or script styles `.pre`, so these boxes only inflate the DOM.
        markup = _NEUTRAL_PRE_SPAN_RE.sub(lambda match: match.group("text"), markup)
        # The active Pygments style leaves names and punctuation unstyled.  Its
        # `.w` rule only colors whitespace, which has no painted glyphs.
        markup = _unwrap_neutral_pygments_spans(markup)
        markup = _WHITESPACE_PYGMENTS_SPAN_RE.sub(
            lambda match: match.group("text"), markup
        )
        markup = _EMPTY_SPAN_RE.sub("", markup)
        markup = _specialize_mathjax_config(markup)

        def _defer_script(match):
            tag = match.group(0)
            if re.search(r"\b(?:defer|async)\b", tag, flags=re.IGNORECASE):
                return tag
            return _append_html_attribute(tag, "defer", "defer")

        markup = external_script_re.sub(_defer_script, markup)
        markup = navigation_bootstrap_re.sub(
            (
                '<script>document.addEventListener("DOMContentLoaded",function(){'
                r"SphinxRtdTheme.Navigation.enable(\1);"
                "},{once:true});</script>"
            ),
            markup,
        )
        # Sphinx emits deprecated bibliography roles and parameter-only inner
        # definition lists.  Repair their semantics without changing layout.
        markup = markup.replace('role="doc-biblioentry"', 'role="listitem"')
        markup = signature_parameter_list_re.sub(
            r'\1<dt class="autolyap-sr-only">Parameters</dt>\n',
            markup,
        )

        def _complete_image(match):
            tag = match.group(0)
            source = _html_attribute(tag, "src")
            css_class = _html_attribute(tag, "class")
            parsed_source = urlparse(source)
            if not parsed_source.netloc:
                asset_path = (html_path.parent / unquote(parsed_source.path)).resolve()
                try:
                    asset_path.relative_to(Path(outdir).resolve())
                except ValueError:
                    asset_path = None
                if asset_path is not None:
                    dimensions = _built_image_dimensions(asset_path)
                    if dimensions is not None:
                        tag = _append_html_attribute(tag, "width", dimensions[0])
                        tag = _append_html_attribute(tag, "height", dimensions[1])
            if "toggler" in css_class.split():
                tag = _append_html_attribute(tag, "loading", "eager")
                tag = _append_html_attribute(tag, "decoding", "sync")
            else:
                tag = _append_html_attribute(tag, "loading", "lazy")
                tag = _append_html_attribute(tag, "decoding", "async")
            return tag

        markup = _HTML_IMAGE_RE.sub(_complete_image, markup)
        html_path.write_text(markup, encoding="utf-8")


def _subset_font(source, destination, codepoints):
    """Create a deterministic WOFF2 subset while retaining outlines and hinting."""
    unicodes = ",".join(f"U+{codepoint:04X}" for codepoint in sorted(codepoints))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fontTools.subset",
            str(source),
            f"--output-file={destination}",
            "--flavor=woff2",
            f"--unicodes={unicodes}",
            "--ignore-missing-unicodes",
            # Keep fontTools' standard shaping features, but do not retain
            # discretionary alternates that the documentation CSS never enables.
            "--notdef-glyph",
            "--notdef-outline",
            "--recommended-glyphs",
            "--drop-tables+=FFTM",
            "--no-recalc-timestamp",
        ],
        check=True,
    )


class _CssUsageParser(HTMLParser):
    """Collect static classes and IDs that generated theme selectors can match."""

    def __init__(self):
        super().__init__()
        self.classes = set()
        self.ids = set()

    def handle_starttag(self, _tag, attrs):
        for name, value in attrs:
            if not value:
                continue
            if name == "class":
                self.classes.update(value.split())
            elif name == "id":
                self.ids.add(value)


_RUNTIME_THEME_CLASSES = {
    # sphinx-rtd-theme navigation and table transformations
    "current",
    "on",
    "shift",
    "shift-up",
    "toctree-expand",
    "wy-table-responsive",
    # Sphinx search and query highlighting
    "highlight-link",
    "highlighted",
    "kind-index",
    "kind-object",
    "kind-text",
    "kind-title",
}
_CSS_CLASS_REFERENCE_RE = re.compile(r"\.([A-Za-z_][\w-]*)")
_CSS_ID_REFERENCE_RE = re.compile(r"#([A-Za-z_][\w-]*)")
_QUOTED_IDENTIFIER_RE = re.compile(
    r"[\"']([A-Za-z_][\w-]*(?:\s+[A-Za-z_][\w-]*)*)[\"']"
)


def _generated_css_usage(outdir):
    """Return selector identifiers present in HTML or authored runtime assets."""
    parser = _CssUsageParser()
    for html_path in Path(outdir).rglob("*.html"):
        parser.feed(html_path.read_text(encoding="utf-8"))

    parser.classes.update(_RUNTIME_THEME_CLASSES)
    runtime_assets = [
        Path(outdir) / "_static" / "custom.css",
        *Path(outdir).joinpath("_static").rglob("*.js"),
    ]
    for asset_path in runtime_assets:
        if not asset_path.is_file():
            continue
        source = asset_path.read_text(encoding="utf-8")
        parser.classes.update(_CSS_CLASS_REFERENCE_RE.findall(source))
        parser.ids.update(_CSS_ID_REFERENCE_RE.findall(source))
        # Capture class names passed as JavaScript string constants, including
        # classList operations that do not contain a CSS-style leading dot.
        for value in _QUOTED_IDENTIFIER_RE.findall(source):
            parser.classes.update(value.split())

    return parser.classes, parser.ids


def _split_selector_branches(tokens):
    """Split a selector prelude on top-level commas without touching functions."""
    branches = []
    branch = []
    for token in tokens:
        if token.type == "literal" and token.value == ",":
            branches.append(branch)
            branch = []
        else:
            branch.append(token)
    branches.append(branch)
    return branches


def _selector_can_match(tokens, used_classes, used_ids):
    """Conservatively reject selectors requiring absent top-level classes/IDs."""
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if (
            token.type == "literal"
            and token.value == "."
            and next_token.type == "ident"
            and next_token.value not in used_classes
        ):
            return False
    return all(
        token.value in used_ids
        for token in tokens
        if token.type == "hash" and getattr(token, "is_identifier", False)
    )


def _prune_css_rules(rules, used_classes, used_ids):
    """Drop only selector branches that cannot match the generated site."""
    retained = []
    grouping_at_rules = {"container", "document", "layer", "media", "supports"}

    for rule in rules:
        if rule.type == "qualified-rule":
            branches = _split_selector_branches(rule.prelude)
            live_branches = [
                branch
                for branch in branches
                if _selector_can_match(branch, used_classes, used_ids)
            ]
            if not live_branches:
                continue
            if len(live_branches) != len(branches):
                selector = ",".join(
                    tinycss2.serialize(branch) for branch in live_branches
                )
                rule.prelude = tinycss2.parse_component_value_list(selector)
            retained.append(rule)
            continue

        if (
            rule.type == "at-rule"
            and rule.content is not None
            and rule.lower_at_keyword in grouping_at_rules
        ):
            nested = tinycss2.parse_rule_list(
                rule.content,
                skip_comments=False,
                skip_whitespace=False,
            )
            nested = _prune_css_rules(nested, used_classes, used_ids)
            rule.content = tinycss2.parse_component_value_list(
                tinycss2.serialize(nested)
            )
        retained.append(rule)

    return retained


def _theme_font_face_family(rule):
    """Return a normalized family name for a top-level ``@font-face`` rule."""
    if (
        rule.type != "at-rule"
        or rule.lower_at_keyword != "font-face"
        or rule.content is None
    ):
        return ""
    declarations = tinycss2.parse_declaration_list(
        rule.content,
        skip_comments=True,
        skip_whitespace=True,
    )
    for declaration in declarations:
        if declaration.type == "declaration" and declaration.lower_name == "font-family":
            return tinycss2.serialize(declaration.value).strip(" \"'").casefold()
    return ""


def _prune_theme_css(outdir, source_theme_css_path):
    """Remove RTD component selectors unused by any generated page or script."""
    theme_css_path = Path(outdir) / "_static" / "css" / "theme.css"
    source_theme_css_path = Path(source_theme_css_path)
    if not theme_css_path.is_file() or not source_theme_css_path.is_file():
        return

    used_classes, used_ids = _generated_css_usage(outdir)
    stylesheet = tinycss2.parse_stylesheet(
        source_theme_css_path.read_text(encoding="utf-8"),
        skip_comments=False,
        skip_whitespace=False,
    )
    # custom.css supplies the active FontAwesome subset and already overrides
    # every rendered Roboto Slab use with the site's Lato family.
    stylesheet = [
        rule
        for rule in stylesheet
        if _theme_font_face_family(rule) not in {"fontawesome", "roboto slab"}
    ]
    stylesheet = _prune_css_rules(stylesheet, used_classes, used_ids)
    theme_css_path.write_text(tinycss2.serialize(stylesheet), encoding="utf-8")


def _prune_generated_stylesheet(outdir, stylesheet_path):
    """Remove selectors that cannot match any generated page or runtime class."""
    stylesheet_path = Path(stylesheet_path)
    if not stylesheet_path.is_file():
        return
    used_classes, used_ids = _generated_css_usage(outdir)
    stylesheet = tinycss2.parse_stylesheet(
        stylesheet_path.read_text(encoding="utf-8"),
        skip_comments=False,
        skip_whitespace=False,
    )
    stylesheet = _prune_css_rules(stylesheet, used_classes, used_ids)
    stylesheet_path.write_text(tinycss2.serialize(stylesheet), encoding="utf-8")


def _minify_generated_stylesheet(stylesheet_path):
    """Remove non-rendering comments and formatting from built CSS."""
    stylesheet_path = Path(stylesheet_path)
    if not stylesheet_path.is_file():
        return
    stylesheet = tinycss2.parse_stylesheet(
        stylesheet_path.read_text(encoding="utf-8"),
        skip_comments=True,
        skip_whitespace=True,
    )
    stylesheet_path.write_text(_serialize_minified_css_rules(stylesheet), encoding="utf-8")


def _compact_css_prelude(tokens):
    """Collapse formatting whitespace while retaining selector combinators."""
    compact = []
    for index, token in enumerate(tokens):
        if token.type != "whitespace":
            compact.append(tinycss2.serialize([token]))
            continue
        previous = next(
            (candidate for candidate in reversed(tokens[:index]) if candidate.type != "whitespace"),
            None,
        )
        following = next(
            (candidate for candidate in tokens[index + 1 :] if candidate.type != "whitespace"),
            None,
        )
        if previous is None or following is None:
            continue
        if (
            previous.type == "literal"
            and previous.value in {",", ">", "+", "~"}
        ) or (
            following.type == "literal"
            and following.value in {",", ">", "+", "~"}
        ):
            continue
        if not compact or compact[-1] != " ":
            compact.append(" ")
    return "".join(compact)


def _serialize_minified_declarations(content):
    """Serialize one declaration block without indentation or redundant separators."""
    declarations = tinycss2.parse_declaration_list(
        content,
        skip_comments=True,
        skip_whitespace=True,
    )
    if any(item.type == "error" for item in declarations):
        return tinycss2.serialize(content).strip()
    serialized = []
    for declaration in declarations:
        if declaration.type != "declaration":
            serialized.append(tinycss2.serialize([declaration]).strip())
            continue
        value = tinycss2.serialize(declaration.value).strip()
        important = "!important" if declaration.important else ""
        serialized.append(f"{declaration.name}:{value}{important}")
    return ";".join(filter(None, serialized))


def _serialize_minified_css_rules(rules):
    """Serialize stylesheet rules recursively without changing declarations."""
    declaration_at_rules = {"font-face", "page", "property", "counter-style"}
    grouping_at_rules = {
        "-webkit-keyframes",
        "container",
        "document",
        "keyframes",
        "layer",
        "media",
        "supports",
    }
    serialized = []
    for rule in rules:
        if rule.type == "qualified-rule":
            prelude = _compact_css_prelude(rule.prelude)
            declarations = _serialize_minified_declarations(rule.content)
            serialized.append(f"{prelude}{{{declarations}}}")
            continue
        if rule.type != "at-rule":
            serialized.append(tinycss2.serialize([rule]).strip())
            continue

        keyword = rule.lower_at_keyword
        prelude = _compact_css_prelude(rule.prelude)
        header = f"@{keyword}{f' {prelude}' if prelude else ''}"
        if rule.content is None:
            serialized.append(f"{header};")
        elif keyword in declaration_at_rules:
            serialized.append(
                f"{header}{{{_serialize_minified_declarations(rule.content)}}}"
            )
        elif keyword in grouping_at_rules:
            nested = tinycss2.parse_rule_list(
                rule.content,
                skip_comments=True,
                skip_whitespace=True,
            )
            serialized.append(f"{header}{{{_serialize_minified_css_rules(nested)}}}")
        else:
            serialized.append(f"{header}{{{tinycss2.serialize(rule.content).strip()}}}")
    return "".join(serialized)


def _write_performance_assets(app, exception):
    """Generate lean font assets and remove unused duplicate build artifacts."""
    if exception is not None or app.builder.format != "html":
        return

    import sphinx_rtd_theme

    outdir = Path(app.outdir)
    _finalize_generated_markup(outdir)
    theme_font_dir = (
        Path(sphinx_rtd_theme.__file__).resolve().parent / "static/css/fonts"
    )
    output_font_dir = outdir / "_static/fonts"

    # The theme also copies legacy font bundles here; no generated CSS references them.
    shutil.rmtree(output_font_dir / "Lato", ignore_errors=True)
    shutil.rmtree(output_font_dir / "RobotoSlab", ignore_errors=True)
    output_font_dir.mkdir(parents=True, exist_ok=True)

    text_codepoints = _font_subset_codepoints(outdir)
    for source_name, output_name in {
        "lato-normal.woff2": "autolyap-lato-normal.woff2",
        "lato-bold.woff2": "autolyap-lato-bold.woff2",
        "lato-normal-italic.woff2": "autolyap-lato-normal-italic.woff2",
        "lato-bold-italic.woff2": "autolyap-lato-bold-italic.woff2",
    }.items():
        _subset_font(
            theme_font_dir / source_name,
            output_font_dir / output_name,
            text_codepoints,
        )
    for source_name, output_name in {
        "lato-normal.woff2": "autolyap-lato-greek-normal.woff2",
        "lato-bold.woff2": "autolyap-lato-greek-bold.woff2",
    }.items():
        _subset_font(
            theme_font_dir / source_name,
            output_font_dir / output_name,
            _font_greek_codepoints(),
        )
    _subset_font(
        theme_font_dir / "fontawesome-webfont.woff2",
        output_font_dir / "autolyap-fontawesome.woff2",
        _fontawesome_subset_codepoints(outdir),
    )
    static_dir = outdir / "_static"
    _prune_theme_css(outdir, theme_font_dir.parent / "theme.css")
    _prune_generated_stylesheet(outdir, static_dir / "pygments.css")

    # Image directives already copied these SVGs to _images; the static copies are unused.
    _minify_generated_stylesheet(static_dir / "css" / "theme.css")
    _minify_generated_stylesheet(static_dir / "custom.css")
    _minify_generated_stylesheet(static_dir / "pygments.css")
    # No generated page references the legacy compatibility shim after filtering.
    (static_dir / "_sphinx_javascript_frameworks_compat.js").unlink(missing_ok=True)
    for unused_font_pattern in ("fontawesome-webfont.*", "Roboto-Slab-*"):
        for unused_font in (static_dir / "css" / "fonts").glob(unused_font_pattern):
            unused_font.unlink()
    shutil.rmtree(outdir / "_sources", ignore_errors=True)
    image_dir = outdir / "_images"
    for static_svg in static_dir.glob("*.svg"):
        if (image_dir / static_svg.name).is_file():
            static_svg.unlink()


def _remove_duplicate_viewport_metatag(context):
    """Let the Read the Docs theme emit the page's single viewport tag."""
    metatags = context.get("metatags")
    if not metatags:
        return
    cleaned = re.sub(
        r'<meta\b(?=[^>]*\bname=["\']viewport["\'])[^>]*?/?>',
        "",
        str(metatags),
        flags=re.IGNORECASE,
    )
    context["metatags"] = type(metatags)(cleaned)


def _as_utc_iso(value):
    """Normalize an ISO datetime string to a UTC ``Z`` representation."""
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@lru_cache(maxsize=None)
def _git_source_dates(source_path_string):
    """Return first and latest Git timestamps for one documentation source."""
    source_path = Path(source_path_string).resolve()
    history = []
    try:
        relative_path = source_path.relative_to(root)
        result = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--format=%cI",
                "--",
                relative_path.as_posix(),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            history = [
                normalized
                for line in result.stdout.splitlines()
                if (normalized := _as_utc_iso(line))
            ]
    except (OSError, subprocess.SubprocessError, ValueError):
        history = []

    if history:
        return history[-1], history[0]

    try:
        timestamp = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return "", ""
    fallback = timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return fallback, fallback


def _document_source_dates(app, docname):
    """Return first-published and latest Git dates for a document source."""
    try:
        source_path = Path(app.env.doc2path(docname))
    except Exception:
        return "", ""
    return _git_source_dates(str(source_path))


_COLLECTION_DOCNAMES = {
    "algorithms",
    "api_reference",
    "contributing",
    "dev/dev_reference",
    "examples",
    "examples/chambolle_pock/index",
    "examples/define_your_own_algorithm/index",
    "examples/douglas_rachford/index",
    "examples/gradient_method/index",
    "examples/heavy_ball/index",
    "examples/nesterov_momentum/index",
    "theory",
    "whats_new",
}

_API_REFERENCE_DOCNAMES = {
    "base_algorithms",
    "concrete_algorithms",
    "function_classes",
    "iteration_dependent_analysis",
    "iteration_independent_analysis",
    "lyapunov_analyses",
    "operator_classes",
    "problem_class",
    "solver_backends",
}


def _infer_schema_type(pagename, root_doc, seo_page):
    """Choose the most specific Schema.org type supported by a page."""
    configured_type = seo_page.get("schema_type")
    if configured_type:
        return configured_type
    if pagename == root_doc:
        return "WebPage"
    if pagename in _COLLECTION_DOCNAMES:
        return "CollectionPage"
    if pagename in _API_REFERENCE_DOCNAMES or pagename.startswith("dev/dev_internal_"):
        return "APIReference"
    return "TechArticle"


def _learning_resource_defaults(pagename, schema_type):
    """Return conservative educational metadata for documentation categories."""
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if "CollectionPage" in schema_types or "WebPage" in schema_types:
        return "", "", ""
    if pagename.startswith("examples/"):
        return "worked example", "Advanced", "Expert"
    if pagename.startswith("theory/"):
        return "technical reference", "Advanced", "Expert"
    if pagename.startswith("contributing/") or pagename == "quick_start":
        return "guide", "Beginner", "Beginner"
    if "APIReference" in schema_types:
        return "API reference", "Intermediate", "Intermediate"
    if pagename.startswith("release_notes/"):
        return "release notes", "Intermediate", "Intermediate"
    return "technical documentation", "Intermediate", "Intermediate"


def _build_breadcrumb_items(context, page_url, baseurl, pagename, root_doc):
    """Build absolute structured breadcrumbs from Sphinx's relative parents."""
    if not page_url or pagename == root_doc:
        return []

    home_url = f"{baseurl}/"
    items = [{"name": "Home", "url": home_url}]
    seen_urls = {home_url}
    for parent in context.get("parents", []):
        link = parent.get("link", "") if hasattr(parent, "get") else ""
        title = parent.get("title", "") if hasattr(parent, "get") else ""
        parent_url = urljoin(page_url, str(link))
        parent_title = _normalize_meta_text(title)
        if not parent_url or not parent_title or parent_url in seen_urls:
            continue
        seen_urls.add(parent_url)
        items.append({"name": parent_title, "url": parent_url})

    current_title = _normalize_meta_text(context.get("title", ""))
    if current_title and page_url not in seen_urls:
        items.append({"name": current_title, "url": page_url})
    return items


def _inject_seo_page_context(app, pagename, templatename, context, doctree):
    """Expose normalized URL and indexability flags to templates."""
    if app.builder.format != "html":
        return

    _remove_duplicate_viewport_metatag(context)

    seo_pages = app.config.html_context.get("seo_pages", {})
    seo_page = seo_pages.get(pagename, {}) if isinstance(seo_pages, dict) else {}
    if not isinstance(seo_page, dict):
        seo_page = {}
    description = seo_page.get("description")
    if description:
        context["seo_page_description"] = _truncate_meta_description(description)
    else:
        context["seo_page_description"] = _extract_auto_page_description(doctree)
        if not context["seo_page_description"]:
            context["seo_page_description"] = _build_fallback_description(
                context.get("title", ""),
                app.config.project,
            )

    seo_default_keywords = app.config.html_context.get("seo_default_keywords", [])
    seo_page_keywords = seo_page.get("keywords")
    if isinstance(seo_page_keywords, list) and seo_page_keywords:
        context["seo_page_keywords"] = seo_page_keywords
    else:
        context["seo_page_keywords"] = _build_fallback_keywords(
            pagename,
            context.get("title", ""),
            seo_default_keywords,
        )

    schema_type = _infer_schema_type(pagename, app.config.root_doc, seo_page)
    resource_type, educational_level, proficiency_level = _learning_resource_defaults(
        pagename, schema_type
    )
    context["seo_schema_type"] = schema_type
    context["seo_learning_resource_type"] = seo_page.get(
        "learning_resource_type", resource_type
    )
    context["seo_educational_level"] = seo_page.get(
        "educational_level", educational_level
    )
    context["seo_proficiency_level"] = seo_page.get(
        "proficiency_level", proficiency_level
    )
    context["seo_dependencies"] = seo_page.get("dependencies", "")
    context["seo_teaches"] = seo_page.get("teaches", [])
    context["seo_citation"] = seo_page.get("citation", {})
    context["seo_word_count"] = (
        len(re.findall(r"\b[\w'-]+\b", doctree.astext())) if doctree is not None else 0
    )

    feature_flags = _collect_page_feature_flags(doctree)
    if pagename.startswith("_modules/") and pagename != "_modules/index":
        # View-code pages are generated outside a source doctree but still need
        # syntax highlighting and copy controls.
        feature_flags["page_has_code_blocks"] = True
    context.update(feature_flags)
    _filter_optional_script_files(
        context,
        page_has_code_blocks=feature_flags["page_has_code_blocks"],
        page_has_proofs=feature_flags["page_has_proofs"],
        page_needs_math_tag_links=feature_flags["page_needs_math_tag_links"],
    )
    _filter_optional_css_files(
        context,
        page_has_code_blocks=feature_flags["page_has_code_blocks"],
    )
    _optimize_content_markup(app, context)

    is_noindex = _is_noindex_docname(pagename, seo_pages)
    context["seo_is_noindex"] = is_noindex
    published_iso, modified_iso = _document_source_dates(app, pagename)
    context["seo_published_iso"] = published_iso
    context["seo_lastmod_iso"] = modified_iso

    baseurl = _normalized_baseurl(app)
    if not baseurl:
        return

    page_url = _docname_page_url(app, pagename, baseurl)
    context["pageurl"] = page_url
    context["seo_page_url"] = page_url
    context["seo_breadcrumb_items"] = (
        []
        if is_noindex
        else _build_breadcrumb_items(
            context,
            page_url,
            baseurl,
            pagename,
            app.config.root_doc,
        )
    )


def _write_sitemap_and_robots(app, exception):
    """Emit sitemap.xml and robots.txt for HTML builds."""
    if exception is not None or app.builder.format != "html":
        return

    baseurl = _normalized_baseurl(app)
    if not baseurl:
        return

    env = app.builder.env
    outdir = Path(app.builder.outdir)
    seo_pages = app.config.html_context.get("seo_pages", {})
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]

    for docname in sorted(env.found_docs):
        if _is_noindex_docname(docname, seo_pages):
            continue
        loc = _docname_page_url(app, docname, baseurl)
        _, lastmod = _document_source_dates(app, docname)
        sitemap_entry = ["  <url>", f"    <loc>{xml_escape(loc)}</loc>"]
        if lastmod:
            sitemap_entry.append(f"    <lastmod>{lastmod}</lastmod>")
        sitemap_entry.append("  </url>")
        sitemap_lines.extend(sitemap_entry)

    sitemap_lines.append("</urlset>")
    (outdir / "sitemap.xml").write_text(
        "\n".join(sitemap_lines) + "\n", encoding="utf-8"
    )

    robots_lines = [
        "User-agent: *",
        "Allow: /",
        "Disallow: /_sources/",
        f"Sitemap: {baseurl}/sitemap.xml",
    ]
    (outdir / "robots.txt").write_text("\n".join(robots_lines) + "\n", encoding="utf-8")


def _patch_bibtex_local_citation_targets():
    """Prefer same-page bibliography entries when resolving repeated cite keys."""
    try:
        import docutils.nodes as docutils_nodes
        import pybtex.plugin as pybtex_plugin
        from sphinxcontrib.bibtex.citation_target import parse_citation_targets
        from sphinxcontrib.bibtex.domain import BibtexDomain, logger as bibtex_logger
        from sphinxcontrib.bibtex.style.referencing import format_references
        from sphinxcontrib.bibtex.style.template import SphinxReferenceInfo
    except Exception:
        return

    if getattr(BibtexDomain.resolve_xref, "_autolyap_local_first_patch", False):
        return

    def _resolve_xref_local_first(
        self,
        env,
        fromdocname,
        builder,
        typ,
        target,
        node,
        contnode,
    ):
        targets = parse_citation_targets(target)
        keys = {target2.key: target2 for target2 in targets}
        citations_by_key = {}
        for citation in self.citations:
            if citation.key not in keys:
                continue
            if self.bibliographies[citation.bibliography_key].list_ != "citation":
                continue

            previous = citations_by_key.get(citation.key)
            if previous is None:
                citations_by_key[citation.key] = citation
                continue

            prev_local = previous.bibliography_key.docname == fromdocname
            curr_local = citation.bibliography_key.docname == fromdocname
            # Prefer local entries; otherwise keep "last one wins" behavior.
            if curr_local or not prev_local:
                citations_by_key[citation.key] = citation

        for key in keys:
            if key not in citations_by_key:
                bibtex_logger.warning(
                    'could not find bibtex key "%s"' % key,
                    location=node,
                    type="bibtex",
                    subtype="key_not_found",
                )

        plaintext = pybtex_plugin.find_plugin("pybtex.backends", "plaintext")()
        references = [
            (
                citation.entry,
                citation.formatted_entry,
                SphinxReferenceInfo(
                    builder=builder,
                    fromdocname=fromdocname,
                    todocname=citation.bibliography_key.docname,
                    citation_id=citation.citation_id,
                    title=(
                        citation.tooltip_entry.text.render(plaintext).replace(
                            "\\url ",
                            "",
                        )
                        if citation.tooltip_entry
                        else None
                    ),
                    pre_text=keys[citation.key].pre,
                    post_text=keys[citation.key].post,
                ),
            )
            for citation in citations_by_key.values()
        ]
        formatted_references = format_references(self.reference_style, typ, references)
        result_node = docutils_nodes.inline(rawsource=target)
        result_node += formatted_references.render(self.backend)
        return result_node

    _resolve_xref_local_first._autolyap_local_first_patch = True
    BibtexDomain.resolve_xref = _resolve_xref_local_first


def setup(app):
    _patch_python_toc_entries()
    _patch_bibtex_local_citation_targets()
    app.connect("html-page-context", _inject_seo_page_context)
    app.connect("doctree-read", _suppress_member_toc_entries)
    app.connect("build-finished", _write_sitemap_and_robots)
    app.connect("build-finished", _write_performance_assets)
    # Defer non-critical scripts to reduce render-blocking time.
    app.add_js_file("copybutton.js", defer="defer")
    app.add_js_file("content_width_toggle.js", defer="defer")
    app.add_js_file("proof_toggle.js", defer="defer")
    app.add_js_file("math_tag_links.js", defer="defer")


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
suppress_warnings = ["bibtex.duplicate_citation"]
