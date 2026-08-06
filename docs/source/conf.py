# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

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
# Pin MathJax for stable glyph rendering across environments.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"

# MathJax macros aligned with Paper/ver_5/commands.tex and Paper/ver_5/preamble.tex.
mathjax3_config = {
    "loader": {
        "load": [],
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


def _optimize_content_image_markup(app, context):
    """Emit image sizing and loading hints before the browser discovers images."""
    body = context.get("body")
    if not body:
        return

    static_dir = Path(app.srcdir) / "_static"
    badge_index = 0

    def _enhance_image(match):
        nonlocal badge_index
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
            if badge_index == 0:
                tag = _append_html_attribute(tag, "fetchpriority", "high")
            badge_index += 1
        else:
            tag = _append_html_attribute(tag, "loading", "lazy")
            tag = _append_html_attribute(tag, "fetchpriority", "low")
        return tag

    optimized_body = _HTML_IMAGE_RE.sub(_enhance_image, str(body))

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
    external_script_re = re.compile(
        r"<script\b(?=[^>]*\bsrc=)[^>]*>", flags=re.IGNORECASE
    )
    navigation_bootstrap_re = re.compile(
        r"<script>\s*jQuery\(function \(\) \{\s*"
        r"SphinxRtdTheme\.Navigation\.enable\((true|false)\);\s*"
        r"\}\);\s*</script>",
        flags=re.IGNORECASE,
    )

    for html_path in Path(outdir).rglob("*.html"):
        markup = html_path.read_text("utf-8")

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
            "--layout-features=*",
            "--glyph-names",
            "--symbol-cmap",
            "--legacy-cmap",
            "--notdef-glyph",
            "--notdef-outline",
            "--recommended-glyphs",
            "--name-IDs=*",
            "--name-legacy",
            "--name-languages=*",
            "--retain-gids",
            "--drop-tables+=FFTM",
            "--no-recalc-timestamp",
        ],
        check=True,
    )


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
    _subset_font(
        theme_font_dir / "fontawesome-webfont.woff2",
        output_font_dir / "autolyap-fontawesome.woff2",
        _fontawesome_subset_codepoints(outdir),
    )

    # Image directives already copied these SVGs to _images; the static copies are unused.
    static_dir = outdir / "_static"
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
    _optimize_content_image_markup(app, context)

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
