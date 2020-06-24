[![Build Status](https://travis-ci.com/QMCSoftware/QMCSoftware.png?branch=master)](https://travis-ci.com/QMCSoftware/QMCSoftware)
[![codecov](https://codecov.io/gh/QMCSoftware/QMCSoftware/branch/master/graph/badge.svg)](https://codecov.io/gh/QMCSoftware/QMCSoftware)
[![Documentation Status](https://readthedocs.org/projects/qmcpy/badge/?version=latest)](https://qmcpy.readthedocs.io/en/latest/?badge=latest)


# Quasi-Monte Carlo Community Software

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 

[Homepage](https://qmcsoftware.github.io/QMCSoftware/) | [GitHub](https://github.com/QMCSoftware/QMCSoftware) | [Read the Docs](https://qmcpy.readthedocs.io/en/latest/) | [PyPI](https://pypi.org/project/qmcpy/)
<hr>

## Installation

~~~
pip install qmcpy
~~~

**For Developers/Contributors**

This package is dependent on the `qmcpy/` directory being on your python path. This is easiest with a virtual environment. For example, using `virtualenv` and `virtualenvwrapper`

~~~
mkvirtualenv qmcpy
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
setvirtualenvproject
add2virtualenv $(pwd)
pip install -r requirements/dev.txt
pip install -e ./
~~~

For `conda` users, 

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

Note that the QRNG C backend files can be explicitely recompiled with

~~~
pip install -e ./
~~~

<hr>

## Documentation 

The QMCPy Read the Docs is located [here](https://qmcpy.readthedocs.io/en/latest/) with HTML, PDF, and EPUB downloads available [here](https://readthedocs.org/projects/qmcpy/downloads/).

Automated project documentation is compiled with [Sphinx](http://www.sphinx-doc.org/). To compile HTML, PDF, or EPUB docs locally into `sphinx/_build/` first install additional requirements 

~~~
pip install -r requirements/dev_docs.txt
~~~

Then setup Sphinx paths (only needs to be run once for initialization)

~~~
make _doc
~~~

Finally, run one of the following three commands

~~~
make doc_html
make doc_pdf
make doc_epub
~~~

To recompile documentation you now only need one of the above three commands. 

<hr>

## QMCPy

The central package including the 5 main components as listed below. Each component is implemented as abstract classes with concrete implementations. For example, the lattice and Sobol sequences are implemented as concrete implementations of the `DiscreteDistribution` abstract class. A complete list of concrete implementations and thorough documentation can be found on the [QMCPy Read the Docs site](https://qmcpy.readthedocs.io/en/latest/algorithms.html). 

- **Stopping Criterion:** determines the number of samples necessary to meet an error tolerence.
- **Integrand:** the function/process whose expected value will be approximated.
- **True Measure:** the distribution to be intergrated over.
- **Discrete Distribution:** a generator of nodes/sequences, that can be either iid (for Monte Carlo) or low-discrepancy (for quasi-Monte Carlo), that mimic a standard distribution.
- **Accumulate Data:** stores information from integration process.

<hr>

## Workouts and Demos

Workouts extensively test and compare the componenets of the the QMCPy package.
Demos, implemented as Jupyter notebooks, demonstrate functionality and uses cases for QMCPy. They often draw from and explore the output of various workouts. 

To run all workouts (~10 min) use the command

~~~
make workout
~~~

<hr>

## Unitests

Combined doctests and fast (<1 sec) / long (<10 sec) unittests can be run with

~~~
make tests
~~~

See the makefile for individual testing commands.

<hr>

## Developers
 
- Sou-Cheng T. Choi
- Fred J. Hickernell
- Michael McCourt
- Jagadeeswaran Rathinavel
- Aleksei Sorokin

<hr>

## Collaborators
- Mike Giles
- Marius Hofert
- Christiane Lemieux
- Dirk Nuyens

<hr>

## Citation

If you find QMCPy helpful in your work, please support us by citing the following work:

 Choi, S.-C. T., Hickernell, F. J., McCourt, M., Rathinavel, J. & Sorokin, A. QMCPy: A quasi-Monte Carlo Python Library. Working. 2020. [https://qmcsoftware.github.io/QMCSoftware/](https://qmcsoftware.github.io/QMCSoftware/).

This work is maintained under the Apache 2.0 License.

<hr>

## References

<b>[1]</b> F.Y. Kuo & D. Nuyens. "Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients - a survey of analysis and implementation",Foundations of Computational Mathematics, 16(6):1631-1696, 2016. ([springer link](https://link.springer.com/article/10.1007/s10208-016-9329-5), [arxiv link](https://arxiv.org/abs/1606.06613))

<b>[2]</b> Fred J. Hickernell, Lan Jiang, Yuewei Liu, and Art B. Owen, "Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling," Monte Carlo and Quasi-Monte Carlo Methods 2012 (J. Dick, F.Y. Kuo, G. W. Peters, and I. H. Sloan, eds.), pp. 105-128, Springer-Verlag, Berlin, 2014. DOI: 10.1007/978-3-642-41095-6_5

<b>[3]</b> Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. Available from http://gailgithub.github.io/GAIL_Dev/

<b>[4]</b> Sou-Cheng T. Choi, "MINRES-QLP Pack and Reliable Reproducible Research via Supportable Scientific Software," Journal of Open Research Software, Volume 2, Number 1, e22, pp. 1-7, 2014.

<b>[5]</b> Sou-Cheng T. Choi and Fred J. Hickernell, "IIT MATH-573 Reliable Mathematical Software" [Course Slides], Illinois Institute of Technology, Chicago, IL, 2013. Available from http://gailgithub.github.io/GAIL_Dev/

<b>[6]</b> Daniel S. Katz, Sou-Cheng T. Choi, Hilmar Lapp, Ketan Maheshwari, Frank Loffler, Matthew Turk, Marcus D. Hanwell, Nancy Wilkins-Diehr, James Hetherington, James Howison, Shel Swenson, Gabrielle D. Allen, Anne C. Elster, Bruce Berriman, Colin Venters, "Summary of the First Workshop On Sustainable Software for Science: Practice and Experiences (WSSSPE1)," Journal of Open Research Software, Volume 2, Number 1, e6, pp. 1-21, 2014.

<b>[7]</b> Fang, K.-T., & Wang, Y. (1994). Number-theoretic Methods in Statistics. London, UK: CHAPMAN & HALL

<b>[8]</b> Lan Jiang, Guaranteed Adaptive Monte Carlo Methods for Estimating Means of Random Variables, PhD Thesis, Illinois Institute of Technology, 2016.

<b>[9]</b> Lluis Antoni Jimenez Rugama and Fred J. Hickernell, "Adaptive multidimensional integration based on rank-1 lattices," Monte Carlo and Quasi-Monte Carlo  Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1411.1966, pp. 407-422.

<b>[10]</b> Kai-Tai Fang and Yuan Wang, Number-theoretic Methods in Statistics, Chapman & Hall, London, 1994.

<b>[11]</b> Fred J. Hickernell and Lluis Antoni Jimenez Rugama, "Reliable adaptive cubature using digital sequences", Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1410.8615 [math.NA], pp. 367-383.

<b>[12]</b> Marius Hofert and Christiane Lemieux (2019). qrng: (Randomized) Quasi-Random Number Generators. R package version 0.0-7. https://CRAN.R-project.org/package=qrng.

<b>[13]</b> Faure, Henri, and Christiane Lemieux. “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases.” Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.

<b>[14]</b> M.B. Giles. 'Multi-level Monte Carlo path simulation'. Operations Research, 56(3):607-617, 2008. http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf.

<b>[15]</b> M.B. Giles. `Improved multilevel Monte Carlo convergence using the Milstein scheme'. 343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008. http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf.

<b>[16]</b> M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'. pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics, de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf

<b>[17]</b> Owen, A. B.A randomized Halton algorithm in R2017. arXiv:1706.02808 [stat.CO]

<hr>

## Sponsors

<img src="sphinx/logo/illinois-institute-of-technology-vector-logo.jpg" alt="IIT logo" height=200px width=400px/>

<img src="sphinx/logo/SigOpt_Logo_Files/Horz/Blue/SigoOpt-Horz-Blue.jpg" alt="SigOpt logo" height=100px width=400px/>

<img src="sphinx/logo/kamakura-corporation-vector-logo.png" alt="Kamakura logo" height=200px width=400px/>
