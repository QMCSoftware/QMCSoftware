QMCPy
=====

The QMCPy framework uses 5 abstract classes that are fleshed out in
concrete implementations. Specifically, a user selects an integrand,
true measure, discrete distribution, and stopping criterion specific to
their Monte Carlo (MC) / quasi-Monte Carlo (qMC) problem. The
:math:`5^{th}` abstract class accumulates data from the stopping
criterion and does not need to be instantiated by the user. The
following blocks give more detailed descriptions of each abstract class
and the available concrete implementations. For specific class names and
parameters see the `QMCPy Documentation
page <https://qmcpy.readthedocs.io/en/latest/algorithms.html>`__.

.. raw:: html

   <hr>

Integrand
---------

The function to integrate.

-  Keister Function:
   :math:`g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)`
-  Custom Function
-  European Option
-  Asian Call Option

   -  stock price at time :math:`jT/d`: :math:`~~~~~~~~~`
      :math:`S(x_j)=S_0\exp\bigl((r-\sigma^2/2)(jT/d)+\sigma\mathcal{B}(t_j)\bigr)`
   -  discounted call payoff
      :math:`= \max\left(\frac{1}{d}\sum_{j=1}^{d} S(x_j)-K\right),\: 0) \,\exp(-rT)`
   -  discounted put payoff
      :math:`= \max\left(K-\frac{1}{d}\sum_{j=1}^{d} S(x_j)\right),\: 0)\,\exp(-rT)`

-  Multilevel Call Options with Milstein Discretization
-  Linear Function: :math:`g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}`

.. raw:: html

   <hr>

True Measure
------------

General measure used to define the integrand.

-  Uniform: :math:`\mathcal{U}(\boldsymbol{a},\boldsymbol{b})`
-  Gaussian: :math:`\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})`
-  Discrete Brownian Motion:
   :math:`\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})`, where
   :math:`\mathsf{\Sigma} = \min(\boldsymbol{t},\boldsymbol{t})^T`,
   :math:`~~~~` :math:`\boldsymbol{t} = (t_1, \ldots, t_d)`
-  Lebesgue
-  Importance sampling
-  Identical to what the discrete distribution mimics

.. raw:: html

   <hr>

Discrete Distribution
---------------------

Sampling nodes.

**Low Discrepancy (LD) nodes**

-  Lattice (base 2):
   :math:`\overset{\text{LD}}{\sim} \mathcal{U}(0,1)^d`
-  Sobol' (base 2): :math:`\overset{\text{LD}}{\sim} \mathcal{U}(0,1)^d`
-  Generalized Halton
-  Korobov

**Independent Identically Distributed (IID) Nodes**

-  IID Standard Uniform:
   :math:`\overset{\text{IID}}{\sim} \mathcal{U}(0,1)^d`
-  IID Standard Gaussian:
   :math:`\overset{\text{IID}}{\sim} \mathcal{N}(\boldsymbol{0}_d,\mathsf{I}_d)`
-  Custom IID Distribution
-  Inverse CDF Sampling
-  Acceptance Rejection Sampling

.. raw:: html

   <hr>

Stopping Criterion
------------------

| The stopping criterion to determine sufficient approximation.
| Has class method ``integrate`` which preforms numerical integration.

**qMC Algorithms**

-  Gauranteed Lattice Cubature
-  Guaranteed Sobol Cubature
-  Multilevel qMC Cubature
-  CLT qMC Cubature (with Replications)

**MC Algorithms**

-  Multilevel MC Cubature
-  Garunteed MC Cubature
-  CLT MC Cubature
