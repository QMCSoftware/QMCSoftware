# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../')) # qmcpy
sys.path.insert(0, os.path.abspath('../algorithms/'))
print("root directory = %s " % os.getcwd())


# -- Project information -----------------------------------------------------

project = u'qmcpy'
copyright = u'2019, Illinois Institute of Technology'
author = 'uFred J. Hickernell, Aleksei Sorokin, Sou-Cheng T. Choi'

# The full version, including alpha/beta/rc tags
release = u'0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
              'sphinx.ext.imgmath',
              'sphinx.ext.napoleon',
              'sphinx.ext.graphviz']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Fork me on GitHub banner
html_theme_options = {
    'font_size': '18px',
    'github_user': 'QMCSoftware',
    'github_repo': 'QMCSoftware',
    'github_button': True,
    'github_banner': True
}


# -- Options for LaTeX output ---------------------------------------------

# Paths  on S.C. Choi's Mac machine for latex
imgmath_latex=r"/Library/TeX/texbin/latex"
imgmath_dvipng=r"/Library/TeX/texbin/dvipng"
#imgmath_font_size = 10


#templates_path = ['_templates']

