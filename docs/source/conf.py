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
    'sphinx_multiversion',  # Support for versioning
]
myst_enable_extensions = [
    "dollarmath",   # Enables $...$ syntax
    "amsmath",      # Enables LaTeX-style math blocks with \[ \] and environments
]

smv_tag_whitelist = r"^v\d+\.\d+.*$"   # Matches tags like v1.0.0, v2.1.1, etc.
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"

templates_path = ['_templates']
exclude_patterns = []


html_theme = "furo"
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "display_gitlab": False,
    "versions": [],  # sphinx-multiversion will populate this
}

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/thatmariia/gaussquad",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/thatmariia/gaussquad",
    "source_branch": "main"
}
