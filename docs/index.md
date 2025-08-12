# QMCPy: Quasi-Monte Carlo Community Software in Python

[![](https://img.shields.io/badge/qmcpy.org-15bfa9)](https://qmcpy.org/)
[![](https://img.shields.io/badge/Docs-6b03fc)](https://QMCSoftware.github.io/QMCSoftware/)
[![](https://img.shields.io/badge/PyPI-fc7303)](https://pypi.org/project/qmcpy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964489.svg)](https://doi.org/10.5281/zenodo.3964489)
[![Tests](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/python-package-conda.yml?query=branch%3Amaster)
[![GitHub stars](https://img.shields.io/github/stars/QMCSoftware/QMCSoftware?style=social)](https://github.com/QMCSoftware/QMCSoftware)
[![](https://img.shields.io/badge/GitHub_Issues-030ffc)](https://github.com/QMCSoftware/QMCSoftware/issues)

[![](https://img.shields.io/badge/Fred_Hickernell's_MCQMC2020_Video_Tutorial-FF0000)](https://www.youtube.com/watch?v=gL8M_7c-YUE)
[![](https://img.shields.io/badge/Aleksei_Sorokin's_PyData_Chicago_Video_Tutorial-FF0000)](https://www.youtube.com/watch?v=bRcKiLA2yBQ)

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: a discrete distribution, a true measure of randomness, an integrand, and a stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications.

To learn more about the framework, please read our [mathematical description of QMCPy software and components](https://qmcsoftware.github.io/QMCSoftware/components).

## Installation

```bash
pip install qmcpy
```

To install from source, please see the [contributing guidelines](https://qmcsoftware.github.io/QMCSoftware/contributing).

## Citation

If you find QMCPy helpful in your work, please support us by citing the following work, which is also available as a [QMCPy BibTex citation](https://github.com/QMCSoftware/QMCSoftware/blob/master/cite_qmcpy.bib)

~~~
Sou-Cheng T. Choi, Fred J. Hickernell, Michael McCourt, Jagadeeswaran Rathinavel, Aleksei G. Sorokin,
QMCPy: A Quasi-Monte Carlo Python Library. 2025.
https://qmcsoftware.github.io/QMCSoftware/
~~~

We maintain [a list of publications on the development and use of QMCPy](https://qmcpy.org/publications/) as well as a [list of select references upon which QMCPy was built](https://qmcsoftware.github.io/QMCSoftware/references).

## For Developers

Want to contribute to QMCPy? Please see our [Guidelines for Contributors](https://qmcsoftware.github.io/QMCSoftware/contributing) which includes instructions on installation for developers, running tests, and compiling documentation.

This software would not be possible without the efforts and support of [QMCPy Developers and Sponsors](https://qmcsoftware.github.io/QMCSoftware/community).

QMCPy is distributed under an [Apache 2.0 License from the Illinois Institute of Technology](https://github.com/QMCSoftware/QMCSoftware/blob/master/LICENSE).



## Quickstart

Note: If the following mathematics is not rendering try using Google Chrome and installing the [Mathjax Plugin for GitHub](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

We will approximate the expected value of the $d$ dimensional Keister integrand [18]

$$g(X)=\pi^{d/2}\cos(\lVert X \rVert)$$

where $X \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}/2)$.

We may choose a Sobol' discrete distribution with a corresponding Sobol' cubature stopping criterion to preform quasi-Monte Carlo integration.

```python
import qmcpy as qp
from numpy import pi, cos, sqrt, linalg
d = 2
dnb2 = qp.DigitalNetB2(d)
gauss_sobol = qp.Gaussian(dnb2, mean=0, covariance=1/2)
k = qp.CustomFun(
  true_measure = gauss_sobol, 
  g = lambda x: pi**(d/2)*cos(linalg.norm(x,axis=1)))
qmc_sobol_algorithm = qp.CubQMCSobolG(k, abs_tol=1e-3)
solution,data = qmc_sobol_algorithm.integrate()
print(data)
```

Running the above code outputs

```
LDTransformData (AccumulateData Object)
    solution        1.808
    error_bound     4.68e-04
    n_total         2^(13)
    time_integrate  0.008
CubQMCSobolG (StoppingCriterion Object)
    abs_tol         0.001
    rel_tol         0
    n_init          2^(10)
    n_max           2^(35)
CustomFun (Integrand Object)
Gaussian (TrueMeasure Object)
    mean            0
    covariance      2^(-1)
    decomp_type     PCA
Sobol (DiscreteDistribution Object)
    d               2^(1)
    dvec            [0 1]
    randomize       LMS_DS
    graycode        0
    entropy         127071403717453177593768120720330942628
    spawn_key       ()
```

A more detailed quickstart can be found in our GitHub repo at `QMCSoftware/demos/quickstart.ipynb` or in [this Google Colab quickstart notebook](https://colab.research.google.com/drive/1CQweuON7jHJBMVyribvosJLW4LheQXBL?usp=sharing). 

We also highly recommend you take a look at [Fred Hickernell's tutorial at the Monte Carlo Quasi-Monte Carlo 2020 Conference](https://media.ed.ac.uk/media/Fred+Hickernell%2C+Illinois+Institute+of+Technology+++Quasi-Monte+Carlo+Software++%28MCQMC+2020%2C+10.08.20%29/1_2k12mwiw) and [the corresponding MCQMC2020 Google Colab notebook.](https://tinyurl.com/QMCPyTutorial)

----

## Community

Please refer to [this document](https://github.com/QMCSoftware/QMCSoftware/blob/develop/community.md) for the key roles in the QMCPy community.

---

## Video Tutorial
Please refer to [this video](https://www.youtube.com/watch?v=bRcKiLA2yBQ) for a quick introduction to QMCPy.
[![Watch the video](https://img.youtube.com/vi/bRcKiLA2yBQ/0.jpg)](https://youtu.be/bRcKiLA2yBQ)

For a more detail introduction refer to [this video](https://www.youtube.com/watch?v=gL8M_7c-YUE).
[![Watch the video](https://img.youtube.com/vi/gL8M_7c-YUE/0.jpg)](https://youtu.be/gL8M_7c-YUE)
