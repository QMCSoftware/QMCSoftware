QMCPy
=====

Integrand
---------

| The function to integrate
| *Abstract class with concrete implementations*

-  Linear: :math:`g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}`
-  Keister:
   :math:`g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)`
-  European Option
-  Asian Call

   -  stock price at time :math:`jT/d`: :math:`~~~~~~~~~`
      :math:`S(x_j)=S_0\exp\bigl((r-\sigma^2/2)(jT/d)+\sigma\mathcal{B}(t_j)\bigr)`
   -  discounted call payoff
      :math:`= \max\left(\frac{1}{d}\sum_{j=1}^{d} S(x_j)-K\right),\: 0) \,\exp(-rT)`
   -  discounted put payoff
      :math:`= \max\left(K-\frac{1}{d}\sum_{j=1}^{d} S(x_j)\right),\: 0)\,\exp(-rT)`

-  Multilevel Call Options
-  QuickConstruct

.. raw:: html

   <hr>

True Measure
------------

| General measure used to define the integrand
| *Abstract class with concrete implementations*

-  Uniform: :math:`\mathcal{U}(\boldsymbol{a},\boldsymbol{b})`
-  Gaussian: :math:`\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})`
-  Discrete Brownian Motion:
   :math:`\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})`, where
   :math:`\mathsf{\Sigma} = \min(\boldsymbol{t},\boldsymbol{t})^T`,
   :math:`~~~~` :math:`\boldsymbol{t} = (t_1, \ldots, t_d)`
-  Lebesgue
-  Identical to what the discrete distribution mimics
-  Importance sampling

.. raw:: html

   <hr>

Discrete Distribution
---------------------

| Sampling nodes IID or LDS (low-discrepancy sequence)
| *Abstract class with concrete implementations*

**Independent Identically Distributed (IID) Nodes**

-  IID Standard Uniform:
   :math:`\overset{\text{IID}}{\sim} \mathcal{U}(0,1)^d`
-  IID Standard Gaussian:
   :math:`\overset{\text{IID}}{\sim} \mathcal{N}(\boldsymbol{0}_d,\mathsf{I}_d)`
-  Custom IID Distribution
-  Inverse CDF Sampling
-  Acceptance Rejection Sampling

**Low Discrepancy (LD) nodes**

-  Lattice (base 2):
   :math:`\overset{\text{LD}}{\sim} \mathcal{U}(0,1)^d`
-  Sobol' (base 2): :math:`\overset{\text{LD}}{\sim} \mathcal{U}(0,1)^d`

.. raw:: html

   <hr>

Stopping Criterion
------------------

| The stopping criterion to determine sufficient approximation
| Has class method ``integrate`` which preforms numerical integration
| *Abstract class with concrete implementations*

**For IID Nodes** :math:`x_i\sim` iid

-  Mean MC (guaranteed)
-  Multilevel MC
-  Central Limit Theorem (CLT)

**For QMC Sequences** :math:`\{x_{r,i}\}_{r=1}^R \sim` ld

-  CubLattice\_g (gauranteed)
-  CubSobol\_g (gauranteed)
-  Multilevel QMC
-  CLT Repeated
