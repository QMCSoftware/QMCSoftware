# utils/latex_macros.py
MARKDOWN_CELL = r"""
$
\newcommand{\vh}{\boldsymbol{h}}
\newcommand{\vt}{\boldsymbol{t}}
\newcommand{\vx}{\boldsymbol{x}}
\newcommand{\vX}{\boldsymbol{X}}
\newcommand{\cf}{\mathcal{F}}
\newcommand{\cu}{\mathcal{U}}
\newcommand{\dif}{\mathrm{d}}
\newcommand{\Ex}{\mathbb{E}}
\DeclareMathOperator{\disc}{disc}
\newcommand{\norm}[2]{{\left \lVert #2 \right \rVert}_{#1}}
$
""".strip()

MATPLOTLIB_PREAMBLE = (
    r"\usepackage{amsmath,amssymb}"
    r"\newcommand{\vh}{\boldsymbol{h}}"
    r"\newcommand{\vt}{\boldsymbol{t}}"
    r"\newcommand{\vx}{\boldsymbol{x}}"
    r"\newcommand{\vX}{\boldsymbol{X}}"
    r"\newcommand{\cf}{\mathcal{F}}"
    r"\newcommand{\cu}{\mathcal{U}}"
    r"\newcommand{\dif}{\mathrm{d}}"
    r"\newcommand{\Ex}{\mathbb{E}}"
    r"\DeclareMathOperator{\disc}{disc}"
    r"\newcommand{\norm}[2]{{\left \lVert #2 \right \rVert}_{#1}}"
)