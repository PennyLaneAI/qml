# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings
import numpy as np
from jinja2 import FileSystemLoader, Environment
import yaml
from pennylane import PennyLaneDeprecationWarning

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------
# General information about the project.

project = "PennyLane"

copyright = "2022, Xanadu Quantum Technologies, Inc."

author = "Xanadu Inc."

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "1.8.5"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx_gallery.gen_gallery",
    "extension",
]


html_baseurl = 'https://pennylane.ai/qml/'

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["demonstrations"],
    # path where to save gallery generated examples
    "gallery_dirs": ["demos"],
    # execute files that match the following filename pattern,
    # and skip those that don't. If the following option is not provided,
    # all example scripts in the 'examples_dirs' folder will be skiped.
    "filename_pattern": r"tutorial",
    "pypandoc": True,
    # first notebook cell in generated Jupyter notebooks
    "first_notebook_cell": (
        "# This cell is added by sphinx-gallery\n"
        "# It can be customized to whatever you like\n"
        "%matplotlib inline"
    ),
    # thumbnail size
    "thumbnail_size": (400, 400),
    'reference_url': {
         # The module you locally document uses None
        'pennylane': "https://docs.pennylane.ai/en/stable/",
    },
    'backreferences_dir'  : 'backreferences',
    'doc_module'          : ('pennylane'),
    'junit': '../test-results/sphinx-gallery/junit.xml',
}

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"

# Remove warnings that occur when generating the the tutorials
warnings.filterwarnings(
    "ignore", category=UserWarning, message=r"Matplotlib is currently using agg"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=r"Timestamps in IBMQ backend"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated.+"
)
warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning,
    message=r"Creating an ndarray from ragged"
)

# Raise PennyLane deprecation warnings as errors
warnings.filterwarnings("error", category=PennyLaneDeprecationWarning)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*venv"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pennylane"
html_theme_path = ["."]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "navbar_active_link": 0,
    "extra_copyrights": [
        "TensorFlow, the TensorFlow logo, and any related marks are trademarks of Google Inc."
    ],
    "google_analytics_tracking_id": "UA-130507810-1",
    "toc_global": False,
    "toc_subset": False,
    "toc_hover": False,
    "relations": False,
    "github_repo": "PennyLaneAI/qml"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/light-slider.css", "css/hubs.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {
#     "**": [
#         "searchbox.html",
#         "globaltoc.html",
#     ]
# }


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "QMLdoc"


# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "pennylane": ("https://docs.pennylane.ai/en/latest/", None),
    "catalyst": ("https://docs.pennylane.ai/projects/catalyst/en/stable", None)
}
