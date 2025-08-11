# utils/latex_macros.py
# Single source of truth for LaTeX macros used in Markdown cells and Matplotlib.

# Core macros (kept identical across Markdown + Matplotlib)
COMMON_MACROS = r"""
\usepackage{amsmath,amssymb,amsfonts,bm}
\newcommand{\vh}{\boldsymbol{h}}
\newcommand{\vt}{\boldsymbol{t}}
\newcommand{\vx}{\boldsymbol{x}}
\newcommand{\vX}{\boldsymbol{X}}
\newcommand{\cf}{\mathcal{F}}
\newcommand{\cu}{\mathcal{U}}
\newcommand{\dif}{\mathrm{d}}
\newcommand{\Ex}{\mathbb{E}}
\DeclareMathOperator{\disc}{disc}
% \norm{x}{2} -> \left\lVert x \right\rVert_{2}
\newcommand{\norm}[2]{{\left \lVert #1 \right \rVert}_{#2}}
""".strip()

# For Matplotlib rcParams["text.latex.preamble"]
MATPLOTLIB_PREAMBLE = COMMON_MACROS

# For Jupyter Markdown cells (paste this cell to define macros in the notebook)
MARKDOWN_CELL = "$$\n" + COMMON_MACROS + "\n$$"