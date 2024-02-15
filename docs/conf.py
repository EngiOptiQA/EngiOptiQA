# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EngiOptiQA'
copyright = '2024, Fabian Key'
author = 'Fabian Key'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Path to the root of your project, relative to the documentation source directory.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
autodoc_mock_imports = ['amplify', 'dimod', 'dwave', 'matplotlib', 'numpy', 'prettytable', 'scipy', 'sympy','tikzplotlib']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
