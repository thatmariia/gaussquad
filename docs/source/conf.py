import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


project = 'gaussquad'
copyright = '2025, Mariia Steeghs-Turchina'
author = 'Mariia Steeghs-Turchina'


extensions = [
    'sphinx.ext.autodoc',   # Automatically document your code from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code,
    'myst_parser',          # Markdown parser
]
myst_enable_extensions = [
    "dollarmath",   # Enables $...$ syntax
    "amsmath",      # Enables LaTeX-style math blocks with \[ \] and environments
]

templates_path = ['_templates']
exclude_patterns = []


html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/thatmariia/gaussquad",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "path_to_docs": "docs",
}