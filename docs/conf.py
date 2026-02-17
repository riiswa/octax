# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Octax"
copyright = "2025, Waris Radji"
author = "Waris Radji"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST parser: use Markdown in docs
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Mock heavy optional imports so autodoc can import octax without them
autodoc_mock_imports = [
    "cv2",
    "pygame",
    "PIL",
    "gymnax",
]

# Napoleon settings for NumPy/Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/octax_logo.png"
html_title = "Octax"
html_short_title = "Octax Documentation"
html_baseurl = "https://riiswa.github.io/octax/"

html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
