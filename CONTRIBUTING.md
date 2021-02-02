# For Contributors

Thank you for you interest in contributing to the QMCPy package. The following sections describe how to get set up as a developer, some helpful commands, and a few suggestions before merging changes. Let us know if you have any trouble by sending an email to [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com).

---

## Installation for Developers

To enable `conda` environments, [install miniconda](https://docs.conda.io/en/latest/miniconda.html). 

Then, setup the `qmcpy` virtual environment

~~~
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
conda env create --file requirements/environment.yml
conda activate qmcpy
pip install -e .
~~~

If setup with our `environment.yml` file fails, you can manually create the environment (minus a few extra dependencies)

~~~
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
conda create -n qmcpy python=3.7.0
conda activate qmcpy
conda install conda-build
conda develop .
pip install -r requirements/dev.txt
pip install -e .
~~~

To check for successful installation, run

~~~
make tests
~~~

Note that the C backend files can be explicitly recompiled with

~~~
pip install -e .
~~~

----

## Documentation 

Automated project documentation is compiled with [Sphinx](http://www.sphinx-doc.org/). To compile HTML, PDF, or EPUB docs locally into `sphinx/_build/` you must install [pandoc](https://pandoc.org/installing.html), a [latex distribution](https://www.latex-project.org/get/), and add additional python requirements with the command

~~~
pip install -r requirements/dev_docs.txt
~~~

Then, use one of the following three commands the compile the documentation

~~~
make doc_html
make doc_pdf
make doc_epub
~~~

----

## Workouts and Demos

Workouts extensively test and compare the components of the QMCPy package. Demos, implemented as Jupyter notebooks, demonstrate functionality and uses cases for QMCPy. They often draw on results from corresponding workouts. 

To run all workouts (~10 min) use the command

~~~
make workout
~~~

----

## Unit Tests

Combined doctests and unittests, both fast (<1 sec) / long (<10 sec), can be run with

~~~
make tests
~~~

See the `makefile` for individual testing commands.

----

## Pull Requests

Pull requests should be made into the `develop` branch, as we try and keep the `master` branch consistent with the current release. 

**For a QMCPy component (generator, algorithm, use case) try to ...**

- incorporate and be consistent with other QMCPy components as much as possible.
- keep naming conventions the same across similar components.  
- develop thorough documentation, including doctests. See `qmcpy/stopping_criterion/cub_qmc_sobol_g.py` as an example.
- create fast and/or long unittests in the `test/` directory. 
- create a workout or demo showcasing your new component, preferably including
    - a connection/comparison to available components. 
    - how the expected cost is realized. 
    - an overview of the relevant mathematics. 
    - figures to illustrate important features.
    - references. 
- consider submitting a blog to put on the [QMCPy blogs site](http://qmcpy.wordpress.com/).

**For a bug fix, try to**

- fix/add doctests and unittests.
- update documentation/references. 