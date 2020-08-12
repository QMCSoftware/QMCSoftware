# For Contributors

Thank you for you interest in contributing to the QMCPy package. The following sections describe how to get set up as a developer, some helpful commands, and a few suggestions before merging changes. Let us know if you have any trouble by sending an email to [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com).

---

## Installation for Developers

This package is dependent on the `qmcpy/` directory being on your python path. This is easiest with a virtual environment.

Setup using `virtualenv` and `virtualenvwrapper`

~~~
mkvirtualenv qmcpy
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
setvirtualenvproject
add2virtualenv $(pwd)
pip install -r requirements/dev.txt
pip install -e ./
~~~

Setup using `conda`

~~~
conda create --name qmcpy python=3.6
conda activate qmcpy
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
pip install -r requirements/dev.txt
pip install -e ./
~~~

To check for successful installation, run

~~~
make tests
~~~

Note that the C backend files can be explicitly recompiled with

~~~
pip install -e ./
~~~

----

## Documentation 

Automated project documentation is compiled with [Sphinx](http://www.sphinx-doc.org/). To compile HTML, PDF, or EPUB docs locally into `sphinx/_build/` you must install [pandoc](https://pandoc.org/installing.html), a [latex distribution](https://www.latex-project.org/get/), and add additional python requirements with the command

~~~
pip install -r requirements/dev_docs.txt
~~~

Then setup Sphinx paths (only needs to be run once for initialization) with the command

~~~
make _doc
~~~

Finally, use one of the following three commands the compile the documentation

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

**For a QMCPy component try to ...**

- incorporate other QMCPy components when possible.
- keep naming conventions the same across similar components.  
- develop thorough documentation, including doctests.
- create unittests. 
- create a workout or demo showcasing your new component, 
    - preferably with a connection/comparison to available components. 
    - preferably showing theoretical complexity is met. 


**For a QMCPy use case, try to ...**

- create a demo that showcases your component, preferably including
    - an overview of the mathematics. 
    - figures.
    - references. 
- develop a potential blog to put on the [QMCPy blogs site](http://qmcpy.wordpress.com/).

**For a bug fix, try to**

- fix/add doctests and unittests.
- update documentation/references. 