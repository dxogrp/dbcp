"""Sphinx configuration for the DBCP documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbcp import __version__  # noqa: E402


def _documentation_series(package_version: str) -> str:
    """Return the ``major.minor`` documentation series for a stable release."""
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", package_version)
    if match is None:
        message = f"DBCP package version must be a canonical stable major.minor.patch release: {package_version!r}"
        raise RuntimeError(message)
    return f"{match.group(1)}.{match.group(2)}"


project = "DBCP"
author = "Hao Zhu"
copyright = "2026, Hao Zhu and DBCP contributors"
package_version = __version__
documentation_series = _documentation_series(package_version)
version = documentation_series
release = documentation_series

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

extlinks = {"example": ("examples/%s.html", "%s")}

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_typehints_format = "short"
autosummary_generate = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = True

# Keep the extension enabled without making strict local builds depend on
# downloading third-party inventories. External projects are linked directly.
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}
nitpicky = True
nitpick_ignore_regex = [
    ("py:class", r"(Constraint|Expression|Iterable|Objective|Parameter|Problem|Variable)"),
    ("py:class", r"cp\..*"),
    ("py:class", r"(collections|cvxpy|numpy|types|typing)\..*"),
]

html_theme = "alabaster"
html_title = f"DBCP {release}"
html_baseurl = f"https://dxogrp.github.io/dbcp/version/{release}/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("version-switcher.js", {"defer": "defer"})]
html_theme_options = {
    "description": "Disciplined biconvex programming in Python",
    "fixed_sidebar": True,
    "github_button": True,
    "github_repo": "dbcp",
    "github_type": "star",
    "github_user": "dxogrp",
    "page_width": "1120px",
    "show_powered_by": False,
    "sidebar_width": "270px",
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
        "versions.html",
    ]
}
html_context = {
    "docs_switcher_url": "https://dxogrp.github.io/dbcp/switcher.json",
}


def _open_example_links_in_new_tab(app, doctree, docname):
    """Open rendered examples separately without affecting other links."""
    del docname
    if app.builder.format != "html":
        return
    for reference in doctree.findall():
        attributes = getattr(reference, "attributes", {})
        if "extlink-example" in attributes.get("classes", ()):
            attributes["target"] = "_blank"
            attributes["rel"] = "noopener"


def setup(app):
    """Register documentation-only rendering hooks."""
    app.connect("doctree-resolved", _open_example_links_in_new_tab)
