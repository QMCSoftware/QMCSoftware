QMCPy
=====

Integrand
---------

| The function to integrate
| *Abstract class with concrete implementations*

-  Linear: :math:`g(\boldsymbol{x}_i) = \sum_{j=1}^{d}(x_{ij})`
-  Keister: :math:`y_i = \pi^{d/2} \, \cos(||\boldsymbol{x}_i||_2)`
-  Asian Call

   -  :math:`S_i(t_j)=S(0)e^{(r-\frac{\sigma^2}{2})t_j+\sigma\mathcal{B}(t_j)}`
   -  discounted call payoff
      :math:`= \max(\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d)-K)\;,\: 0)*e^{-rT}`
   -  discounted put payoff
      :math:`= \max(K-\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d))\;,\: 0)*e^{-rT}`

-  QuickConstruct

.. raw:: html

   <hr>

True Measure
------------

| General measure used to define the integrand
| *Abstract class with concrete implementations*

-  Uniform: :math:`\mathcal{U}(a,b)`
-  Gaussian: :math:`\mathcal{N}(\mu,\sigma^2)`
-  Brownian Motion:
   :math:`\mathcal{B}(t_j)=B(t_{j-1})+Z_j\sqrt{t_j-t_{j-1}} \;` for
   :math:`\;Z_j \sim \mathcal{N}(0,1)`
-  Lebesgue
-  Identity Transform

.. raw:: html

   <hr>

Discrete Distribution
---------------------

| Sampling nodes IID or LDS (low-discrepancy sequence)
| *Abstract class with concrete implementations*

-  IID Standard Uniform:
   :math:`x_j \overset{iid}{\sim} \mathcal{U}(0,1)`
-  IID Standard Gaussian:
   :math:`x_j \overset{iid}{\sim} \mathcal{N}(0,1)`
-  Custom IID Distribution
-  Lattice (base 2): :math:`x_j \overset{lds}{\sim} \mathcal{U}(0,1)`
-  Sobol (base 2): :math:`x_j \overset{lds}{\sim} \mathcal{U}(0,1)`
-  Inverse CDF Sampling
-  Acceptance Rejection Sampling

.. raw:: html

   <hr>

Stopping Criterion
------------------

| The stopping criterion to determine sufficient approximation
| Has class method ``integrate`` which preforms numerical integration
| *Abstract class with concrete implementations*

**For IID Nodes** :math:`x_i\sim` iid

-  Central Limit Theorem (CLT)
-  MeanMC\_g (gauranteed)

**For QMC Sequences** :math:`\{x_{r,i}\}_{r=1}^R \sim` lds

-  CLT Repeated
-  CubLattice\_g (gauranteed)
-  CubSobol\_g (gauranteed)

.. raw:: html

   <hr>

Accumulate Data Class
---------------------

| Stores data values of corresponding stopping criterion procedure
| *Abstract class with concrete implementations*

-  Mean Variance Data
-  Mean Variance Repeated Data
-  Cubature Data
