[![Build Status](https://travis-ci.com/QMCSoftware/QMCSoftware.png?branch=master)](https://travis-ci.com/QMCSoftware/QMCSoftware)
[![codecov](https://codecov.io/gh/QMCSoftware/QMCSoftware/branch/master/graph/badge.svg)](https://codecov.io/gh/QMCSoftware/QMCSoftware)
[![Documentation Status](https://readthedocs.org/projects/qmcpy/badge/?version=latest)](https://qmcpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964490.svg)](https://doi.org/10.5281/zenodo.3964490)

# Quasi-Monte Carlo Community Software

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications.

<center><img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/qmcpy_logo.png?raw=true" alt="QMCPy logo" height=200px width=200px/>

[Homepage](https://qmcsoftware.github.io/QMCSoftware/) | [GitHub](https://github.com/QMCSoftware/QMCSoftware) | [Read the Docs](https://qmcpy.readthedocs.io/en/latest/) | [PyPI](https://pypi.org/project/qmcpy/) | [Blogs](http://qmcpy.wordpress.com/) | [Contributing](https://github.com/QMCSoftware/QMCSoftware/blob/master/CONTRIBUTING.md) | [Issues](https://github.com/QMCSoftware/QMCSoftware/issues) | [Citations](https://github.com/QMCSoftware/QMCSoftware/blob/master/citations.md)</center>

----

## Installation

~~~
pip install qmcpy
~~~

----

## The QMCPy Framework

The central package including the 5 main components as listed below. Each component is implemented as abstract classes with concrete implementations. For example, the lattice and Sobol' sequences are implemented as concrete implementations of the `DiscreteDistribution` abstract class. An overview of implemented componenets and some of the underlying mathematics is available in the [QMCPy README](https://github.com/QMCSoftware/QMCSoftware/blob/master/qmcpy/README.md).  A complete list of concrete implementations and thorough documentation can be found in the [QMCPy Read the Docs](https://qmcpy.readthedocs.io/en/latest/algorithms.html) .

- **Stopping Criterion:** determines the number of samples necessary to meet an error tolerance.
- **Integrand:** the function/process whose expected value will be approximated.
- **True Measure:** the distribution to be integrated over.
- **Discrete Distribution:** a generator of nodes/sequences, that can be either IID (for Monte Carlo) or low-discrepancy (for quasi-Monte Carlo), that mimic a standard distribution.
- **Accumulate Data:** stores and updates data used in the integration process.

----

## Quickstart

Note: If the following mathematics is not rendering try using Google Chrome and installing the [Mathjax Plugin for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

Will will find the exptected value of the Keister integrand [18]

$$g(\boldsymbol{x})=\pi^{d/2}\cos(||\boldsymbol{x}||)$$

integrated over a $d$ dimensional Gaussian true measure with variance $1/2$

$$\mathcal{X} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}/2).$$

We may choose a Sobol' discrete distribution with a corresponding Sobol' cubature stopping criterion to preform quasi-Monte Carlo numerical integration.

```python
import qmcpy as qp
from numpy import pi, cos, sqrt, linalg
d = 2
s = qp.Sobol(d)
g = qp.Gaussian(s, covariance=1./2)
k = qp.CustomFun(g, lambda x: pi**(d/2)*cos(linalg.norm(x,axis=1)))
cs = qp.CubQMCSobolG(k, abs_tol=1e-3)
solution,data = cs.integrate()
print(data)
```

A more detailed quickstart can be found in our GitHub repo at `QMCSoftware/demos/quickstart.ipynb` or in [this Google Colab notebook](https://colab.research.google.com/drive/1CQweuON7jHJBMVyribvosJLW4LheQXBL?usp=sharing).

----

## Developers
 
- Sou-Cheng T. Choi
- Fred J. Hickernell
- Michael McCourt
- Jagadeeswaran Rathinavel
- Aleksei Sorokin

----

## Collaborators

- Mike Giles
- Marius Hofert
- Pierre L’Ecuyer
- Christiane Lemieux
- Dirk Nuyens
- Art Owen

----

## Sponsors

Illinois Tech
--------------


   <p style="height:30x">
     <a href="https://www.iit.edu">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/illinois-institute-of-technology-vector-logo.jpg?raw=true"/ width="300" height="150">
     </a>
   </p>

Kamakura Corporation
---------------------


   <p style="height:30x">
     <a href="http://www.kamakuraco.com">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/kamakura-corporation-vector-logo.png?raw=true" width="300" height="150"/>
     </a>
   </p>


SigOpt, Inc.
------------


   <p>
     <a href="https://sigopt.com">
       <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/sphinx/logo/SigOpt_Logo_Files/Horz/Blue/SigoOpt-Horz-Blue.jpg?raw=true" width="300" height="100"/>
     </a>
   </p>

