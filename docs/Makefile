# Minimal makefile for Sphinx documentation

PYTHON ?= $(shell if command -v python >/dev/null 2>&1; then echo python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
SPHINXOPTS =
SPHINXBUILD = $(PYTHON) -m sphinx
SOURCEDIR = source
BUILDDIR = build
REQFILE = requirements.txt

.PHONY: help clean deps check-deps dirhtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

deps:
	@$(PYTHON) -m pip install -r "$(REQFILE)"

check-deps:
	@$(PYTHON) -c "import numpy, sphinx, sphinx_rtd_theme, myst_parser, sphinxcontrib.bibtex; from sphinx.domains.python._object import PyObject" >/dev/null 2>&1 || \
	( \
		echo "Missing docs dependencies."; \
		echo "Run: make -C docs deps"; \
		exit 1; \
	)

dirhtml: check-deps
	@SPHINX_VERSION=$$($(PYTHON) -c "import sphinx; print(sphinx.__version__)"); \
	DOCTREEDIR="$(BUILDDIR)/doctrees-dirhtml-$$SPHINX_VERSION"; \
	$(SPHINXBUILD) -M dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" -d "$$DOCTREEDIR" $(SPHINXOPTS)
