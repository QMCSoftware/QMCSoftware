[![GitHub stars](https://img.shields.io/github/stars/QMCSoftware/QMCSoftware?style=social)](https://github.com/QMCSoftware/QMCSoftware)
[![Documentation Status](https://readthedocs.org/projects/qmcpy/badge/?version=latest)](https://qmcpy.readthedocs.io/en/latest/?badge=latest)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964489.svg)](https://doi.org/10.5281/zenodo.3964489) [![Tests](https://github.com/QMCSoftware/QMCSoftware/workflows/QMCPy_CI/badge.svg)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/python-package-conda.yml)
[![codecov](https://codecov.io/gh/QMCSoftware/QMCSoftware/branch/master/graph/badge.svg)](https://app.codecov.io/gh/QMCSoftware/QMCSoftware)

If you like this project, please ⭐ star it using the button in the upper-right corner of the repository!

If you use QMCPy in your work, please support us by citing the following works:

~~~

1. S.-C. T. Choi, F. J. Hickernell, R. Jagadeeswaran, M. McCourt, and A. Sorokin. QMCPy: A quasi-Monte
Carlo Python library (versions 1–1.6.1), 2025.  doi: 10.5281/zenodo.3964489. URL: https://qmcsoftware.github.io/QMCSoftware/.

2. S.-C. T. Choi, F. J. Hickernell, R. Jagadeeswaran, M. J. McCourt, and A. G. Sorokin. Quasi-Monte
Carlo software. In A. Keller, editor, Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Oxford,
England, August 2020, Springer Proceedings in Mathematics and Statistics, pages 23–50. Springer,
Cham, 2022.  URL: https://arxiv.org/abs/2102.07833

~~~

BibTex citations are available [here](./cite_qmcpy.bib).

# Quasi-Monte Carlo Community Software

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 

<img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/docs/assets/logos/qmcpy_logo.png?raw=true" alt="QMCPy logo" height=200px width=200px/>

[Homepage](https://qmcsoftware.github.io/QMCSoftware/) ~ [Article](https://arxiv.org/abs/2102.07833) ~ [GitHub](https://github.com/QMCSoftware/QMCSoftware) ~ [Read the Docs](https://qmcpy.readthedocs.io/en/latest/) ~ [PyPI](https://pypi.org/project/qmcpy/) ~ [Blogs](http://qmcpy.wordpress.com/) ~ [Contributing](https://github.com/QMCSoftware/QMCSoftware/blob/master/CONTRIBUTING.md) ~ [Issues](https://github.com/QMCSoftware/QMCSoftware/issues)

----

## Installation

~~~
pip install qmcpy
~~~

----

## The QMCPy Framework

The central package including the 5 main components as listed below. Each component is implemented as abstract classes with concrete implementations. For example, the lattice and Sobol' sequences are implemented as concrete implementations of the `DiscreteDistribution` abstract class. A complete list of concrete implementations and thorough documentation can be found in the [QMCPy Read the Docs](https://qmcpy.readthedocs.io/en/latest/algorithms.html).

- **Stopping Criterion:** determines the number of samples necessary to meet an error tolerance.
- **Integrand:** the function/process whose expected value will be approximated.
- **True Measure:** the distribution to be integrated over.
- **Discrete Distribution:** a generator of nodes/sequences that can be either IID (for Monte Carlo) or low-discrepancy (for quasi-Monte Carlo), that mimic a standard distribution.
- **Accumulate Data:** stores and updates data used in the integration process.

----

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


## References

<b>[1]</b> F. Y. Kuo and D. Nuyens. "Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients - a survey of analysis and implementation," Foundations of Computational Mathematics, 16(6):1631-1696, 2016. ([springer link](https://link.springer.com/article/10.1007/s10208-016-9329-5), [arxiv link](https://arxiv.org/abs/1606.06613))

<b>[2]</b> Fred J. Hickernell, Lan Jiang, Yuewei Liu, and Art B. Owen, "Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling," Monte Carlo and Quasi-Monte Carlo Methods 2012 (J. Dick, F.Y. Kuo, G. W. Peters, and I. H. Sloan, eds.), pp. 105-128, Springer-Verlag, Berlin, 2014. DOI: 10.1007/978-3-642-41095-6_5

<b>[3]</b> Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, GAIL: Guaranteed Automatic Integration Library (Version 2.3.1) [MATLAB Software], 2020. Available from [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).

<b>[4]</b> Sou-Cheng T. Choi, "MINRES-QLP Pack and Reliable Reproducible Research via Supportable Scientific Software," Journal of Open Research Software, Volume 2, Number 1, e22, pp. 1-7, 2014.

<b>[5]</b> Sou-Cheng T. Choi and Fred J. Hickernell, "IIT MATH-573 Reliable Mathematical Software" [Course Slides], Illinois Institute of Technology, Chicago, IL, 2013. Available from [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).

<b>[6]</b> Daniel S. Katz, Sou-Cheng T. Choi, Hilmar Lapp, Ketan Maheshwari, Frank Loffler, Matthew Turk, Marcus D. Hanwell, Nancy Wilkins-Diehr, James Hetherington, James Howison, Shel Swenson, Gabrielle D. Allen, Anne C. Elster, Bruce Berriman, Colin Venters, "Summary of the First Workshop On Sustainable Software for Science: Practice and Experiences (WSSSPE1)," Journal of Open Research Software, Volume 2, Number 1, e6, pp. 1-21, 2014.

<b>[7]</b> Fang, K.-T., and Wang, Y. (1994). Number-theoretic Methods in Statistics. London, UK: CHAPMAN & HALL

<b>[8]</b> Lan Jiang, Guaranteed Adaptive Monte Carlo Methods for Estimating Means of Random Variables, PhD Thesis, Illinois Institute of Technology, 2016.

<b>[9]</b> Lluis Antoni Jimenez Rugama and Fred J. Hickernell, "Adaptive multidimensional integration based on rank-1 lattices," Monte Carlo and Quasi-Monte Carlo  Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1411.1966, pp. 407-422.

<b>[10]</b> Kai-Tai Fang and Yuan Wang, Number-theoretic Methods in Statistics, Chapman & Hall, London, 1994.

<b>[11]</b> Fred J. Hickernell and Lluis Antoni Jimenez Rugama, "Reliable adaptive cubature using digital sequences," Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium, April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1410.8615 [math.NA], pp. 367-383.

<b>[12]</b> Marius Hofert and Christiane Lemieux (2019). qrng: (Randomized) Quasi-Random Number Generators. R package version 0.0-7. [https://CRAN.R-project.org/package=qrng](https://CRAN.R-project.org/package=qrng).

<b>[13]</b> Faure, Henri, and Christiane Lemieux. “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases,” Mathematics and Computers in Simulation 161 (2019): 13–22. 

<b>[14]</b> M. B. Giles. "Multi-level Monte Carlo path simulation," Operations Research, 56(3):607-617, 2008. [http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf](http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf).

<b>[15]</b> M. B. Giles. "Improved multilevel Monte Carlo convergence using the Milstein scheme," 343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008. [http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf](http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf).

<b>[16]</b> M. B. Giles and B. J. Waterhouse. "Multilevel quasi-Monte Carlo path simulation," pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics, de Gruyter, 2009. [http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf](http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf).

<b>[17]</b> Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]

<b>[18]</b> B. D. Keister, Multidimensional Quadrature Algorithms,  'Computers in Physics', 10, pp. 119-122, 1996.

<b>[19]</b> L’Ecuyer, Pierre & Munger, David. (2015). LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules. ACM Transactions on Mathematical Software. 42. 10.1145/2754929. 

<b>[20]</b> Fischer, Gregory & Carmon, Ziv & Zauberman, Gal & L’Ecuyer, Pierre. (1999). Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators. Operations Research. 47. 159-164. 10.1287/opre.47.1.159. 

<b>[21]</b> I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, Russian Academy of Sciences, Moscow (1992).

<b>[22]</b> Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 2011. 10.1002/wilm.10056. 

<b>[23]</b> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

<b>[24]</b> S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, SIAM J. Sci. Comput. 30, 2635-2654 (2008).

<b>[25]</b> Paul Bratley and Bennett L. Fox. 1988. Algorithm 659: Implementing Sobol's quasirandom sequence generator. ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. DOI:https://doi.org/10.1145/42288.214372

<b>[26]</b> P. L'Ecuyer, P. Marion, M. Godin, and F. Puchhammer, "A Tool for Custom Construction of QMC and RQMC Point Sets," Monte Carlo and Quasi-Monte Carlo Methods 2020.

<b>[27]</b> P Kumaraswamy, A generalized probability density function for double-bounded random processes. J. Hydrol. 46, 79–88 (1980).

<b>[28]</b> D Li, Reliable quasi-Monte Carlo with control variates. Master’s thesis, Illinois Institute of Technology (2016)

<b>[29]</b> D.H. Bailey, J.M. Borwein, R.E. Crandall, Box integrals, Journal of Computational and Applied Mathematics, Volume 206, Issue 1, 2007, Pages 196-208, ISSN 0377-0427, https://doi.org/10.1016/j.cam.2006.06.010.

<b>[30]</b> Art B. Owen.Monte Carlo theory, methods and examples. 2013.

----

## Sponsors

- **[Illinois Tech](https://www.iit.edu)**

  <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/docs/assets/logos/illinois-institute-of-technology-vector-logo.jpg?raw=true" width="300" height="150">

- **[Kamakura Corporation](http://www.kamakuraco.com), acquired by [SAS Institute Inc.](https://www.sas.com) in June 2022**

  <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/docs/assets/logos/kamakura-corporation-vector-logo.png?raw=true" width="300" height="150"/>

- **[SigOpt, Inc.](https://sigopt.com)**

  <img src="https://github.com/QMCSoftware/QMCSoftware/blob/master/docs/assets/logos/SigOpt_Logo_Files/Horz/Blue/SigoOpt-Horz-Blue.jpg?raw=true" width="300" height="100"/>
