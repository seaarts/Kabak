# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Setting the project path ------------------------------------------------
import os
import sys

# Load custom style file
# html_style = "style.css"

sys.path.insert(0, os.path.abspath("../../kabak/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Kabak"
copyright = "2023, Sander Aarts"
author = "Sander Aarts"
release = "00.00.01"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",  # create to-do boxes
    "sphinx.ext.intersphinx",  # add links to other docs
    "sphinx.ext.todo",  # enable to-do boxes
]

templates_path = ["_templates"]
exclude_patterns = []

# To-Do
todo_include_todos = True

# -- Exclude pattersns -------------------------------------------------------
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_favicon = r"http://aim-bigfoot.uzh.ch/~docs/path/to/icon/favicon.png"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
