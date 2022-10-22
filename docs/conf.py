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

from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "mrinversion"
copyright = "2020, Deepansh J. Srivastava"
author = "Deepansh J. Srivastava"

path = os.path.split(__file__)[0]
# get version number from the file
with open(os.path.join(path, "../mrinversion/__init__.py")) as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            __version__ = after_keyword.strip()[1:-1]

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxjp.themes.basicstrap",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
]

autosummary_generate = True

# ---------------------------------------------------------------------------- #
#                               Plot directive config                          #
# ---------------------------------------------------------------------------- #
plot_html_show_source_link = False
plot_rcparams = {
    "font.size": 10,
    "font.weight": "light",
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
}

# ---------------------------------------------------------------------------- #
#                               Sphinx Gallery config                          #
# ---------------------------------------------------------------------------- #

# filter sphinx matplotlib warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# numfig config
numfig = True
numfig_secnum_depth = 1
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

# math
math_number_all = True

# sphinx gallery config
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],  # path to example scripts
    "remove_config_comments": True,
    "gallery_dirs": ["galley_examples"],  # path to gallery generated output
    "within_subsection_order": FileNameSortKey,
    "subsection_order": ExplicitOrder(
        [
            "../examples/synthetic",
            "../examples/sideband",
            "../examples/MAF",
            "../examples/relaxation",
        ]
    ),
    "reference_url": {
        # The module you locally document uses None
        "mrinversion": None,
    },
    "first_notebook_cell": (
        "# This cell is added by sphinx-gallery\n\n"
        "%matplotlib inline\n\n"
        "import mrinversion\n"
        "print(f'You are using mrinversion v{mrinversion.__version__}')"
    ),
    # "binder": {
    #     # Required keys
    #     "org": "DeepanshS",
    #     "repo": "mrinversion",
    #     "branch": "master",
    #     "binderhub_url": "https://mybinder.org",
    #     "dependencies": "../requirements.txt",
    #     # Optional keys
    #     "filepath_prefix": "docs/_build/html",
    #     "notebooks_dir": "../../notebooks",
    #     "use_jupyter_lab": True,
    # },
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "csdmpy": ("https://csdmpy.readthedocs.io/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

copybutton_prompt_text = ">>> |\\$ |\\[\\d*\\]: |\\.\\.\\.: |[.][.][.] "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------------------- #

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toc-tree document.
master_doc = "index"

# autodoc mock modules
autodoc_mock_imports = []

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Some html_theme options are 'alabaster', 'bootstrap', 'sphinx_rtd_theme',
# 'classic', 'basicstrap'
html_theme = "basicstrap"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # Set the lang attribute of the html tag. Defaults to 'en'
    "lang": "en",
    # Disable showing the sidebar. Defaults to 'false'
    "nosidebar": False,
    # Show header searchbox. Defaults to false. works only "nosidebar=True",
    "header_searchbox": False,
    # Put the sidebar on the right side. Defaults to false.
    "rightsidebar": False,
    # Set the width of the sidebar. Defaults to 3
    "sidebar_span": 3,
    # Fix navbar to top of screen. Defaults to true
    "nav_fixed_top": True,
    # Fix the width of the sidebar. Defaults to false
    "nav_fixed": True,
    # Set the width of the sidebar. Defaults to '900px'
    "nav_width": "300px",
    # Fix the width of the content area. Defaults to false
    "content_fixed": False,
    # Set the width of the content area. Defaults to '900px'
    "content_width": "900px",
    # Fix the width of the row. Defaults to false
    "row_fixed": False,
    # Disable the responsive design. Defaults to false
    "noresponsive": False,
    # Disable the responsive footer relbar. Defaults to false
    "noresponsiverelbar": False,
    # Disable flat design. Defaults to false.
    # Works only "bootstrap_version = 3"
    "noflatdesign": False,
    # Enable Google Web Font. Defaults to false
    "googlewebfont": False,
    # Set the URL of Google Web Font's CSS.
    # Defaults to 'http://fonts.googleapis.com/css?family=Text+Me+One'
    # "googlewebfont_url": "http://fonts.googleapis.com/css?family=Roboto",  # NOQA
    # Set the Style of Google Web Font's CSS.
    # Defaults to "font-family: 'Text Me One', sans-serif;"
    # "googlewebfont_style": u"font-family: 'Roboto' Regular;",  # font-size: 1.5em",
    # Set 'navbar-inverse' attribute to header navbar. Defaults to false.
    "header_inverse": True,
    # Set 'navbar-inverse' attribute to relbar navbar. Defaults to false.
    "relbar_inverse": True,
    # Enable inner theme by Bootswatch. Defaults to false
    "inner_theme": False,
    # Set the name of inner theme. Defaults to 'bootswatch-simplex'
    # "inner_theme_name": "bootswatch-simplex",
    # Select Twitter bootstrap version 2 or 3. Defaults to '3'
    "bootstrap_version": "3",
    # Show "theme preview" button in header navbar. Defaults to false.
    "theme_preview": False,
    # Set the Size of Heading text. Defaults to None
    # "h1_size": "3.0em",
    # "h2_size": "2.6em",
    # "h3_size": "2.2em",
    # "h4_size": "1.8em",
    # "h5_size": "1.9em",
    # "h6_size": "1.1em",
}


# Theme options
html_logo = "_static/mrinversion.png"
html_style = "style.css"
html_title = f"mrinversion:doc v{__version__}"
html_last_updated_fmt = ""
# html_logo = "mrinversion"
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html"],
    "using/windows": ["searchbox.html", "windowssidebar.html"],
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "Mrinversion doc"

# -- Options for LaTeX output ------------------------------------------------
latex_engine = "xelatex"
# latex_logo = "_static/csdmpy.png"
latex_show_pagerefs = True

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "9pt",
    "fontenc": r"\usepackage[utf8]{inputenc}",
    "geometry": r"\usepackage[vmargin=2.5cm, hmargin=2cm]{geometry}",
    # "fncychap": r"\usepackage[Rejne]{fncychap}",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{amsfonts, amsmath, amssymb, mathbbol}
        \usepackage{graphicx}
        \usepackage{setspace}
        \singlespacing

        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \fancyhf{}
        \fancyhead[L]{
            \ifthenelse{\isodd{\value{page}}}{ \small \nouppercase{\leftmark} }{}
        }
        \fancyhead[R]{
            \ifthenelse{\isodd{\value{page}}}{}{ \small \nouppercase{\rightmark} }
        }
        \fancyfoot[CO, CE]{\thepage}
    """,
    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "mrinversion.tex", "mrinversion Documentation", author, "manual")
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "mrinversion", "mrinversion Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Mrinversion",
        "Mrinversion Documentation",
        author,
        "Mrinversion",
        "Statistical learning of tensor distribution from NMR anisotropic spectra",
        "Miscellaneous",
    )
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html", "_static/style.css"]


def setup(app):
    app.add_css_file("style.css")
