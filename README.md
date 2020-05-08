[![Build Status](https://travis-ci.com/QMCSoftware/QMCSoftware.png?branch=master)](https://travis-ci.com/QMCSoftware/QMCSoftware)
[![codecov](https://codecov.io/gh/QMCSoftware/QMCSoftware/branch/master/graph/badge.svg)](https://codecov.io/gh/QMCSoftware/QMCSoftware)
[![Documentation Status](https://readthedocs.org/projects/qmcpy/badge/?version=latest)](https://qmcpy.readthedocs.io/en/latest/?badge=latest)

# Quasi-Monte Carlo Community Software

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 


<hr>

##  Project 

Homepage: [https://qmcsoftware.github.io/QMCSoftware/](https://qmcsoftware.github.io/QMCSoftware/) 

Code repostiory:  [https://github.com/QMCSoftware/QMCSoftware](https://github.com/QMCSoftware/QMCSoftware)

<hr>

## Citation

If you find QMCPy helpful in your work, please support us by citing the following work:

Fred J. Hickernell, Sou-Cheng T. Choi, and Aleksei Sorokin, “QMC  Community Software.” Python software, 2019. Work in progress. Available from [https://github.com/QMCSoftware/QMCSoftware](https://github.com/QMCSoftware/QMCSoftware).

<hr>

## Documentation 

PDF and EPUB ocumentation of QMCPy is available for download at  Read the Docs website
[https://readthedocs.org/projects/qmcpy/downloads/](https://readthedocs.org/projects/qmcpy/downloads/).

In addition, we have HTML documentation at [https://qmcpy.readthedocs.io/en/latest/](https://qmcpy.readthedocs.io/en/latest/)

<hr>

## Developers
 
- Sou-Cheng T. Choi
- Fred J. Hickernell
- Aleksei Sorokin

<hr>

## Contributors

- Michael McCourt

<hr>

## Acknowledgment 

We thank Dirk Nuyens for fruitful discussions related to Magic Point Shop.


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

<hr>

## Sponsors

<img src="./python_prototype/sphinx/logo/illinois-institute-of-technology-vector-logo.jpg" alt="IIT logo"/>

<img src="./python_prototype/sphinx/logo/kamakura-corporation-vector-logo.png" alt="Kamakura logo"/>

<img src="./python_prototype/sphinx/logo/SigOpt_Logo_Files/Horz/Blue/SigoOpt-Horz-Blue.jpg" alt="SigOpt logo"/>

