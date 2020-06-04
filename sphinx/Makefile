# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Clean out documentation files to restart
clean:
	@rm -fr $(BUILDDIR)/*
	@echo "QMCPy clean finished. Files in $(BUILDDIR) are removed."

# Create HTML documentation
html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "QMCPy build finished. The HTML pages are in $(BUILDDIR)/html."

# Create epub documentation
epub:
	@$(SPHINXBUILD) -M epub $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) -D extensions=sphinx.ext.imgmath

# Create PDF documentation
latex:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "QMCPy build finished. The LATEX and PDF files are in $(BUILDDIR)/latex."
