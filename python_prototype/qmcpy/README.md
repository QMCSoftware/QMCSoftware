# Quasi-Monte Carlo Community Software
### Python 3

<hr>

## Overview
Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 

<hr>

## Integrand Class
<b>Summary:</b> The function to integrate<br>
<b>Concrete Classes:</b>
- Linear: $\:\: y_i = \sum_{j=0}^{d-1}(x_{ij})$
- Keister: $\:\: y_i = \pi^{d/2} * \cos(||x_i||_2)$
- Asian Call

<hr>

## True Measure
<b>Summary:</b> General measure used to define the integrand <br>
<b>Available Measures:</b>
- Uniform:$\:\: \mathcal{U}(a,b)$
- Gaussian: $\:\: \mathcal{N}(\mu,\sigma^2)$
- Brownian Motion: $\:\: \mathcal{B}(t)$

<hr>

## Discrete Distribution Class
<b>Summary:</b> The distribution dictating sampling points of the function.<br>
<b>Concrete Classes:</b>
- IID Standard Uniform: $\:\: x_1,x_2,...,x_n \: (iid) \sim   \mathcal{U}(0,1)$
- IID Standard Gaussian: $\:\: x_1,x_2,...,x_n \: (iid) \sim   \mathcal{N}(0,1)$
- Lattice (base 2): $\:\: x_1,x_2,...,x_n  \sim   \mathcal{U}(0,1)$
- Sobol (base 2): $\:\: x_1,x_2,...,x_n  \sim   \mathcal{U}(0,1)$

<hr>

## Stopping Criterion Class
<b>Summary:</b> The stopping criterion to determine sufficient approximation<br>
<b>Concrete Classes:</b>
- Central Limit Theorem (CLT) based procedure for iid samples
- CLT Repeated based procedure for non-iid samples

<hr>   

## Accumulate Data Class
<b>Summary:</b> The data values to track throughout the integration<br>
<b>Concrete Classes:</b>
- Mean Variance Data (Controlled by CLT Stopping Criterion)
- Mean Variance Data Repeated (Controlled by CLT Repeated Stopping Criterion)

<hr>

## Integrate Method
<b>Summary:</b> Preforms integration of the Integrand by generating samples from the Discrete Distribution and transforming them to imitate the True Measure until the Stopping Criterion is met.
<b>Args:</b>
- Integrand object
- True Measure object
- Discrete Distribution object
- Stopping Criterion object

<hr>
<hr>

## Contributors 
Dirk Nuyens Magic Point Shop 
- qmcpy/distribution/magic_point_shop
- F.Y. Kuo & D. Nuyens. Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients - a survey of analysis and implementation, Foundations of Computational Mathematics, 16(6):1631-1696, 2016. ([springer link](https://link.springer.com/article/10.1007/s10208-016-9329-5), [arxiv link](https://arxiv.org/abs/1606.06613))

<hr>

## Developers
This package was originally developed in MATLAB by Fred J. Hickernell.<br>
Translated to Python by Sou-Cheng T. Choi and Aleksei Sorokin.

If you find Qmcpy helpful in your work, please support us by citing the
following work:

Fred J. Hickernell, Sou-Cheng T. Choi, and Aleksei Sorokin, “QMC  Community Software.” MATLAB and Python 3
software, 2019. Work in progress. Available on [GitHub](https://github.com/QMCSoftware/QMCSoftware)