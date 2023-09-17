# For Contributors

Thank you for your interest in contributing to the QMCPy package!

Please submit **pull requests** to the `develop` branch and **issues** using a template from `.github/ISSUE_TEMPLATE/`

If you develop a new component please consider writing a blog for [qmcpy.org](https://qmcpy.org)

Join team communications by reaching out to us at [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com)


## Installation 

In a git enabled terminal (e.g. [bash](https://gitforwindows.org/) for Windows) with [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and C compilers enabled (Windows users may need to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)), run

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
make tests_no_docker
~~~

Optionally, you may install [Docker](https://www.docker.com/products/docker-desktop/) and then run all tests with 

~~~
make tests
~~~

After installing [Pandoc](https://pandoc.org/installing.html) and [LaTeX](https://www.latex-project.org/get/), documentation may be compiled into your preferred format with

```
doc_html
doc_pdf
doc_epub
```

Demos may be run by opening [Jupyter](https://jupyter.org/) using

~~~
jupyter notebook
~~~ 

and then navigating to the desired file in the `demo/` directory. Some of our developers prefer [JupyterLab](https://jupyter.org/), which may be installed with

~~~
pip install jupyterlab
~~~ 

and then run with

~~~
jupyter-lab
~~~

The [Developers Tools](https://qmcpy.org/references-for-python-and-mathematical-software-development/) page on [qmcpy.org](https://qmcpy.org) documents additional tools we have found helpful for mathematical software development and presentation. 

## Visual Studio Code Tips and Tricks

[VSCode](https://code.visualstudio.com) is the IDE of choice for many of our developers. Here we compile some helpful notes regarding additional setup for VSCode. 

- Run `CMD`+`p` then `> Python: Select Interpreter` then select the `('qmcpy')` choice from the dropdown to link the qmcpy environment into your workspace. Now when you open a terminal, your command line should read `(qmcpy) username@...` which indicates the qmcpy environment has been automatically activated. Also, when debugging the qmcpy environment will be automatically used. 
- Go to `File` and click `Save Workspace as...` to save a qmcpy workspace for future development.

### Useful Extensions
- Python
- Jupyter
- Markdown Preview Enhanced
- eps-preview, which requires
    - Postscript Language
    - pdf2svg
- Git Graph
- Code Spell Checker
