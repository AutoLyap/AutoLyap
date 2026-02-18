# Minimal makefile for Sphinx documentation

PYTHON ?= python
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
	@$(PYTHON) -c "import sphinx, myst_parser, sphinxcontrib.bibtex" >/dev/null 2>&1 || \
	( \
		echo "Missing docs dependencies."; \
		echo "Run: make -C docs deps"; \
		exit 1; \
	)

dirhtml: check-deps
	@$(SPHINXBUILD) -M dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
