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
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_material

# -- Project information -----------------------------------------------------

project = 'pytwoway'
copyright = '2020, Thibaut Lamadon'
author = 'Thibaut Lamadon'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx.ext.autosummary',
                'sphinx.ext.autosectionlabel',
                'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['']


# The theme to use for HTML and HTML Help pages. Taken from
# https://github.com/statsmodels/statsmodels/blob/master/docs/source/conf.py
extensions.append('sphinx_material')
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = 'sphinx_material'
html_title = project
html_short_title = project
# material theme options (see theme.conf for more information)

base_url = 'https://github.com/tlamadon/pytwoway'
html_theme_options = {
    'base_url': base_url,
    'repo_url': 'https://github.com/tlamadon/pytwoway',
    'repo_name': 'tlamadon',
    'globaltoc_depth': 3,
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
    'color_primary': 'indigo',
    'color_accent': 'blue',
    'nav_title': 'pytwoway v{0}'.format(release),
    'master_doc': False, # Gets rid of unnecessary bar at top that appears when scrolling up
    'nav_links': [],
    # 'heroes': {'index': 'statistical models, hypothesis tests, and data '
    #                     'exploration',
    #            'examples/index': 'examples and tutorials to get started with '
    #                              'statsmodels'},
    "version_dropdown": True,
    # "version_json": "_static/versions.json",
}

language = 'en'
html_last_updated_fmt = ''

# Fix to left sidebar not working with Material
# Source: https://github.com/bashtage/sphinx-material/issues/30
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}
