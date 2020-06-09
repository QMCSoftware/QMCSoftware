Integration Examples using QMCPy package
========================================

In this demo, we show how to use ``qmcpy`` for performing numerical
multiple integration of two built-in integrands, namely, the Keister
function and the Asian put option payoff. To start, we import the
``qmcpy`` module and the function ``arrange()`` from ``numpy`` for
generating evenly spaced discrete vectors in the examples.

.. code:: ipython3

    from qmcpy import *
    from numpy import arange

Keister Example
---------------

We recall briefly the mathematical definitions of the Keister function,
the Gaussian measure, and the Sobol distribution:

-  Keister integrand: :math:`y_j = \pi^{d/2} \cos(||x_j||_2)`

-  Gaussian true measure: :math:`\mathcal{N}(0,\frac{1}{2})`

-  Sobol discrete distribution:
   :math:`x_j \overset{lds}{\sim} \mathcal{U}(0,1)`

The following code snippet integrates a three-dimensional Keister
function numerically by creating instances of ``qmcpy``'s built-in
classes, ``Keister``, ``Gaussian``, ``Sobol`` and ``CubSobol_g``, as
inputs to the function ``integrate()``.

.. code:: ipython3

    dim = 3
    distribution = Sobol(dimension=dim, scramble=True, seed=7, backend='QRNG')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubSobol_g(integrand,abs_tol=.05).integrate()
    print(data)


.. parsed-literal::

    Solution: 2.1718         
    Keister (Integrand Object)
    Sobol (DiscreteDistribution Object)
    	dimension       3
    	scramble        1
    	seed            7
    	backend         qrng
    	mimics          StdUniform
    	graycode        0
    Gaussian (TrueMeasure Object)
    	distrib_name    Sobol
    	mean            0
    	covariance      0.5000
    CubSobol_g (StoppingCriterion Object)
    	abs_tol         0.0500
    	rel_tol         0
    	n_init          1024
    	n_max           34359738368
    CubatureData (AccumulateData Object)
    	n_total         1024
    	solution        2.1718
    	r_lag           4
    	time_integrate  0.0027
    


Arithmetic-Mean Asian Put Option: Single Level
----------------------------------------------

In this example, we want to estimate the payoff of an European Asian put
option that matures at time :math:`T`. The key mathematical entities are
defined as follows:

-  Stock price at time :math:`t_j := jT/d` for :math:`j=1,\dots,d` is a
   function of its initial price :math:`S(0)`, interest rate :math:`r`,
   and volatility :math:`\sigma`:
   :math:`S(t_j) = S(0)e^{\left(r-\frac{\sigma^2}{2}\right)t_j + \sigma\mathcal{B}(t_j)}`

-  Discounted put option payoff is defined as the difference of a fixed
   strike price :math:`K` and the arithmetic average of the underlying
   stock prices at :math:`d` discrete time intervals in :math:`[0,T]`:
   :math:`max \left(K-\frac{1}{d}\sum_{j=1}^{d} S(t_j), 0 \right) e^{-rT}`

-  Brownian motion true measure:
   :math:`\mathcal{B}(t_j) = B(t_{j-1}) + Z_j\sqrt{t_j-t_{j-1}} \;` for
   :math:`\;Z_j \sim \mathcal{N}(0,1)`

-  Lattice discrete distribution:
   :math:`\:\: x_j \overset{lds}{\sim} \mathcal{U}(0,1)`

.. code:: ipython3

    distribution = Lattice(dimension=64, scramble=True, seed=7, backend='GAIL')
    measure = BrownianMotion(distribution)
    integrand = AsianCall(
        measure = measure,
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0.01,
        mean_type = 'arithmetic')
    solution,data = CubLattice_g(integrand, abs_tol=.05).integrate()
    print(data)


.. parsed-literal::

    Solution: 6.2744         
    AsianCall (Integrand Object)
    	volatility      0.5000
    	start_price     30
    	strike_price    25
    	interest_rate   0.0100
    	mean_type       arithmetic
    	dimensions      64
    	dim_fracs       0
    Lattice (DiscreteDistribution Object)
    	dimension       64
    	scramble        1
    	seed            7
    	backend         gail
    	mimics          StdUniform
    BrownianMotion (TrueMeasure Object)
    	distrib_name    Lattice
    	time_vector     [ 0.016  0.031  0.047 ...  0.969  0.984  1.000]
    	mean_shift_is   0
    CubLattice_g (StoppingCriterion Object)
    	abs_tol         0.0500
    	rel_tol         0
    	n_init          1024
    	n_max           34359738368
    CubatureData (AccumulateData Object)
    	n_total         4096
    	solution        6.2744
    	r_lag           4
    	time_integrate  0.0541
    


Arithmetic-Mean Asian Put Option: Multi-Level
---------------------------------------------

This example is similar to the last one except that we use Gile's
multi-level method for estimation of the option price. The main idea can
be summarized as follows:

:math:`Y_0 = 0`

:math:`Y_1 = \text{ Asian option monitored at } t = [\frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1]`

:math:`Y_2 = \text{ Asian option monitored at } t= [\frac{1}{16}, \frac{1}{8}, ... , 1]`

:math:`Y_3 = \text{ Asian option monitored at } t= [\frac{1}{64}, \frac{1}{32}, ... , 1]`

:math:`Z_1 = \mathbb{E}[Y_1-Y_0] + \mathbb{E}[Y_2-Y_1] + \mathbb{E}[Y_3-Y_2] = \mathbb{E}[Y_3]`

The total run time for this example is about one-third of that for the
last example.

.. code:: ipython3

    distribution = IIDStdGaussian(seed=7)
    measure = BrownianMotion(distribution)
    integrand = AsianCall(measure,
            volatility = 0.5,
            start_price = 30,
            strike_price = 25,
            interest_rate = 0.01,
            mean_type = 'arithmetic',
            multi_level_dimensions = [4,16,64])
    solution,data = CLT(integrand, abs_tol=.05).integrate()
    print(data)


.. parsed-literal::

    Solution: 6.2533         
    AsianCall (Integrand Object)
    	volatility      0.5000
    	start_price     30
    	strike_price    25
    	interest_rate   0.0100
    	mean_type       arithmetic
    	dimensions      [ 4 16 64]
    	dim_fracs       [ 0.000  4.000  4.000]
    IIDStdGaussian (DiscreteDistribution Object)
    	dimension       64
    	seed            7
    	mimics          StdGaussian
    BrownianMotion (TrueMeasure Object)
    	distrib_name    IIDStdGaussian
    	time_vector     [ 0.016  0.031  0.047 ...  0.969  0.984  1.000]
    	mean_shift_is   0
    CLT (StoppingCriterion Object)
    	inflate         1.2000
    	alpha           0.0100
    	abs_tol         0.0500
    	rel_tol         0
    	n_init          1024
    	n_max           10000000000
    MeanVarData (AccumulateData Object)
    	levels          3
    	solution        6.2533
    	n               [332598  28362   2847]
    	n_total         366879
    	confid_int      [ 6.204  6.302]
    	time_integrate  0.1189
    


