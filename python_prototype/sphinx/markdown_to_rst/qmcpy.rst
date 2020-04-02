.. contents::
   :depth: 1
..

QMCPy
=====

Integrand
---------

| The function to integrate
| *Abstract class with concrete implementations*

-  Linear: :math:`\:\: y_i = \sum_{j=0}^{d-1}(x_{ij})`
-  Keister: :math:`\:\: y_i = \pi^{d/2} \, \cos(||\boldsymbol{x}_i||_2)`
-  Asian Call

   -  :math:`S_i(t_j)=S(0)e^{(r-\frac{\sigma^2}{2})t_j+\sigma\mathcal{B}(t_j)}`
   -  discounted call payoff
      :math:`= \max(\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d)-K)\;,\: 0)`
   -  discounted put payoff
      :math:`= \max(K-\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d))\;,\: 0)`

-  QuickConstruct

--------------

True Measure
------------

| General measure used to define the integrand
| *Abstract class with concrete implementations*

-  Uniform: :math:`\:\: \mathcal{U}(a,b)`
-  Gaussian: :math:`\:\: \mathcal{N}(\mu,\sigma^2)`
-  Brownian Motion:
   :math:`\:\: \mathcal{B}(t_j)=B(t_{j-1})+Z_j\sqrt{t_j-t_{j-1}} \;` for
   :math:`\;Z_j \sim \mathcal{N}(0,1)`
-  Lebesgue

--------------

Discrete Distribution
---------------------

| Sampling nodes IID or LDS (low-discrepancy sequence)
| *Abstract class with concrete implementations*

-  IID Standard Uniform:
   :math:`\:\: x_j \overset{iid}{\sim} \mathcal{U}(0,1)`
-  IID Standard Gaussian:
   :math:`\:\: x_j \overset{iid}{\sim} \mathcal{N}(0,1)`
-  Lattice (base 2):
   :math:`\:\: x_j \overset{lds}{\sim} \mathcal{U}(0,1)`
-  Sobol (base 2): :math:`\:\: x_j \overset{lds}{\sim} \mathcal{U}(0,1)`

--------------

Stopping Criterion
------------------

| The stopping criterion to determine sufficient approximation
| Has class method ``integrate`` which preforms numerical integration
| *Abstract class with concrete implementations*

**For IID Nodes** :math:`x_i\sim` iid

-  Central Limit Theorem (CLT)
-  MeanMC_g (gauranteed)

**For QMC Sequences** :math:`\{x_{r,i}\}_{r=1}^R \sim` lds

-  CLT Repeated
-  CubLattice_g (gauranteed)

--------------

Accumulate Data Class
---------------------

| Stores data values of corresponding stopping criterion procedure
| *Abstract class with concrete implementations*

-  Mean Variance Data
-  Mean Variance Repeated Data
-  Cubature Data
