# Quasi-Monte Carlo Community Software


## Overview
Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 

<hr>

## Function Class
<b>Summary:</b> The function to integrate<br>
<b>Concrete Classes:</b>
- Linear
- Keister
- Asian Call

<hr>

## Distribution Class
<b>Summary:</b> The distribution dictating sampling points of the function.<br>
<b>Concrete Classes:</b>
- Independent Identically Distributed
- QuasiRandom

<hr>

## Accumulate Data Class
<b>Summary:</b> The data values to track throughout the integration<br>
<b>Concrete Classes:</b>
- Mean Variance
- Mean Variance Repeated

<hr>

## Stopping Criterion Class
<b>Summary:</b> The stopping criterion to determine sufficient approximation<br>
<b>Concrete Classes:</b>
- Central Limit Theorem
- Central Limit Theorem Repeated

<hr>   

## Measure
<b>Summary:</b> Specifies the components of a general measure used to define an integration problem or a sampling method<br>
<b>Available Measures:</b>
- STDUniform
- STDGaussian
- IIDZeroMeanGaussian
- BrownianMotion
- Lattice
- Sobol

<hr>

## Integrate Method
<b>Summary:</b> Preforms integration given Function's', Distribution, Data, and a Stopping Criterion 

<hr>

## Developers
This package was originally developed in MATLAB by Fred J. Hickernell.<br>
Translated to Python by Sou-Cheng T. Choi and Aleksei Sorokin.

### Tips
It may be helpful to create a virtual environment for this project.<br>
Packages and versions are compiled in requirements.txt.<br>
In order for the software to find packages, make sure .../qmcpy/ is in your path.



