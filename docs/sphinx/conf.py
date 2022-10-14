# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


# -- Project information -----------------------------------------------------
import os

project = 'CVCUDA'
copyright = '2022, NVIDIA.'
author = 'NVIDIA'
version = 'PreAlpha'
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'docs/manuals/py/**']

#source_parsers = { '.md': 'recommonmark.parser.CommonMarkParser',}

extensions=['recommonmark']


source_suffix = {'.rst': 'restructuredtext', '.md':'markdown'}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_rtd_theme"
html_logo = os.path.join('content', 'nv_logo.png')

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#000000',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': False,
    # 'navigation_depth': 10,
    'sidebarwidth': 12,
    'includehidden': True,
    'titles_only': False
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_favicon = os.path.join('content', 'nv_icon.png')

html_static_path = ['templates']

html_last_updated_fmt = ''

html_js_files = [
    'pk_scripts.js',
]

def setup(app):
    app.add_css_file('custom.css')
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for breathe --------------------------------------------------

# Enable the breathe extension
extensions.append('breathe')
extensions.append('exhale')

# Set up the default project for breathe extension
breathe_default_project = 'cvcuda'


# -- Options for sphinx_rtd_theme -----------------------------------------

# Enable the sphinx_rtd_theme extension
extensions.append('sphinx_rtd_theme')

# Enable the sphinx.ext.todo extension
extensions.append('sphinx.ext.todo')

# -- Extension configuration -------------------------------------------------

doxygen_config = """
INPUT                = ../../src/include
EXCLUDE             += ../../../tests
EXCLUDE_PATTERNS     = *.md *.txt
ENABLE_PREPROCESSING = YES
WARN_IF_UNDOCUMENTED = NO
USE_M
"""

doxygen_conf_extra = """
INLINE_SIMPLE_STRUCTS = YES
TYPEDEF_HIDES_STRUCT = YES
EXPAND_ONLY_PREDEF = YES
"""

doxygen_predefined = [
    "NVCV_PUBLIC=",
    "NVCV_API_VERSION_IS(x,y)=0",
    "NVCV_API_VERSION_AT_LEAST(x,y)=1",
    "NVCV_API_VERSION_AT_MOST(x,y)=0"
]

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "_exhale_api",
    "rootFileName":          "cvcuda_api.rst",
    "doxygenStripFromPath":  "../../src/include",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "Library API",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": False,
    "fullToctreeMaxDepth": 1,
    "minifyTreeView": False,
    "contentsDirectives": False,
    "exhaleDoxygenStdin":    "INPUT = ../../src/include"
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'
