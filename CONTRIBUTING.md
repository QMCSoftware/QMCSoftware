# Contributing

Thank you for your interest in contributing to the QMCPy package!

Please submit **pull requests** to the `develop` branch and **issues** using a template from `.github/ISSUE_TEMPLATE/`

If you develop a new component please consider writing a blog for [qmcpy.org](https://qmcpy.org)

Join team communications by reaching out to us at [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com)

## Installation

In a git enabled terminal (e.g. [bash](https://gitforwindows.org/) for Windows) with [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and C compilers enabled (Windows users may need to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)), run

~~~bash
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
conda create --name qmcpy python=3.12
conda activate qmcpy
pip install -e .[dev]
~~~

While `dev` contains the most complete set of install dependencies, a number of other install dependency groups can be found in our `pyproject.toml` file. If running in the `zsh` terminal you may need to use

~~~bash
pip install -e ".[dev]"
~~~

## Tests

Doctests and unittests take a few minute to run with

~~~bash
make tests_no_docker
~~~

Optionally, you may install [Docker](https://www.docker.com/products/docker-desktop/) and then run all tests with

~~~bash
make tests
~~~

## System Dependencies for Booktests (Developers Only)

Some demonstration notebooks use LaTeX-based rendering (e.g., Matplotlib with `text.usetex=True`), which requires certain **OS-level LaTeX font packages**. These are *not* Python dependencies and therefore cannot be installed via `pip` or listed in `pyproject.toml`.

If you plan to run the full booktest suite (`make booktests_no_docker`), please install the following system packages:

```bash
sudo apt-get update
sudo apt-get install -y cm-super texlive-latex-recommended

Please see the targets in the makefile for more granular control over tests.

## Documentation

To compile documentation, run

```bash 
make doc
```

To download PDF documentation, go to the "Printable Docs" header in the documentation, press Ctrl-p to print, and then choose to save the PDF to your preferred location.

## Demos

Demos are Jupyter notebooks which may be launched using the command

~~~bash
jupyter-lab
~~~

## Other Developer Tools

The [Developers Tools](https://qmcpy.org/references-for-python-and-mathematical-software-development/) page on [qmcpy.org](https://qmcpy.org) documents additional tools we have found helpful for mathematical software development and presentation.

## VSCode Tips

[VSCode](https://code.visualstudio.com) (Visual Studio Code) is the IDE of choice for many of our developers. Here we compile some helpful notes regarding additional setup for VSCode.

- Run `CMD`+`p` then `> Python: Select Interpreter` then select the `('qmcpy')` choice from the dropdown to link the qmcpy environment into your workspace. Now when you open a terminal, your command line should read `(qmcpy) username@...` which indicates the qmcpy environment has been automatically activated. Also, when debugging the qmcpy environment will be automatically used.
- Go to `File` and click `Save Workspace as...` to save a `qmcpy` workspace for future development.

Some VSCode extension we found useful include

- Python
- Jupyter
- Markdown Preview Enhanced
- eps-preview, which requires
    - Postscript Language
    - pdf2svg
- Git Graph
- Code Spell Checker
