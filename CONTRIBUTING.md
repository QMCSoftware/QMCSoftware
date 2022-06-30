# For Contributors

Thank you for your interest in contributing to the QMCPy package!

Please submit **pull requests** to the `develop` branch and **issues** using a template from `.github/ISSUE_TEMPLATE/`

If you develop a new component please consider writing a blog for [qmcpy.org](https://qmcpy.org)

Join team communications by reaching out to us at [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com)


## Installation 

In a git enabled terminal (e.g. [bash](https://gitforwindows.org/) for Windows) with [conda](https://docs.conda.io/en/latest/miniconda.html) installed and C compilers enabled (Windows users may want to install [Cygwin](https://www.cygwin.com) or [MinGW](https://www.mingw-w64.org) and may find this [tutorial from Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp) helpful), run

~~~
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
conda env create --file environment.yml
conda activate qmcpy
pip install -e .
~~~

Doctests and unittests take a few minute to run with

~~~
make tests
~~~

After installing [Pandoc](https://pandoc.org/installing.html) and [$\LaTeX$](https://www.latex-project.org/get/), documentation may be compiled into your preferred format with

```
doc_html
doc_pdf
doc_epub
```

See the `makefile` for more commands and the [developers page](https://qmcpy.org/references-for-python-and-mathematical-software-development/) on [qmcpy.org](https://qmcpy.org) for more tools
