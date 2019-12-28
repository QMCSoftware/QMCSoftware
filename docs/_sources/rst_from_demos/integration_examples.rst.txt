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
function numerically by creating instances of ``qmcpy``\ ’s built-in
classes, ``Keister``, ``Gaussian``, ``Sobol`` and ``CLTRep``, as inputs
to the function ``integrate()``.

.. code:: ipython3

    dim = 3
    integrand = Keister(dim)
    true_measure = Gaussian(dim, variance=1 / 2)
    discrete_distrib = Sobol(rng_seed=7)
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=0.05)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    print(data)


.. parsed-literal::

    Solution: 2.1716         
    Keister (Integrand Object)
    Sobol (Discrete Distribution Object)
    	mimics          StdUniform
    	rng_seed        7
    	backend         pytorch
    Gaussian (True Measure Object)
    	dimension       3
    	mu              0
    	sigma           0.707
    CLTRep (Stopping Criterion Object)
    	abs_tol         0.050
    	rel_tol         0
    	n_max           1073741824
    	inflate         1.200
    	alpha           0.010
    MeanVarDataRep (AccumData Object)
    	n               128
    	n_total         128
    	confid_int      [ 2.164  2.179]
    	time_total      0.010
    	r               16
    


European Arithmetic-Mean Asian Put Option: Single Level
-------------------------------------------------------

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

    time_vec = [arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]
    
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure,
                          volatility = 0.5,
                          start_price = 30,
                          strike_price = 25,
                          interest_rate = 0.01,
                          mean_type = 'arithmetic')
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=0.05)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    print(data)


.. parsed-literal::

    Solution: 6.2595         
    AsianCall (Integrand Object)
    	volatility      0.500
    	start_price     30
    	strike_price    25
    	interest_rate   0.010
    	mean_type       arithmetic
    	exercise_time   1
    Lattice (Discrete Distribution Object)
    	mimics          StdUniform
    	rng_seed        7
    BrownianMotion (True Measure Object)
    	dimension       64
    	time_vector     [ 0.016  0.031  0.047 ...  0.969  0.984  1.000]
    CLTRep (Stopping Criterion Object)
    	abs_tol         0.050
    	rel_tol         0
    	n_max           1073741824
    	inflate         1.200
    	alpha           0.010
    MeanVarDataRep (AccumData Object)
    	n               2048
    	n_total         2048
    	confid_int      [ 6.257  6.262]
    	time_total      0.359
    	r               16
    


European Arithmetic-Mean Asian Put Option: Multi-Level
------------------------------------------------------

This example is similar to the last one except that we use Gile’s
multi-level method for estimation of the option price. The main idea can
be summarized as follows:

:math:`Y_0 = 0`

:math:`Y_1 = \mbox{ Asian option monitored at } t = [\frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1]`

:math:`Y_2 = \mbox{ Asian option monitored at } t= [\frac{1}{16}, \frac{1}{8}, ... , 1]`

:math:`Y_3 = \mbox{ Asian option monitored at } t= [\frac{1}{64}, \frac{1}{32}, ... , 1]`

:math:`Z_1 = \mathbb{E}[Y_1-Y_0] + \mathbb{E}[Y_2-Y_1] + \mathbb{E}[Y_3-Y_2] = \mathbb{E}[Y_3]`

The total run time for this example is about one-third of that for the
last example.

.. code:: ipython3

    time_vec = [arange(1 / 4, 5 / 4, 1 / 4),
                arange(1 / 16, 17 / 16, 1 / 16),
                arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]
    
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure,
                          volatility = 0.5,
                          start_price = 30,
                          strike_price = 25,
                          interest_rate = 0.01,
                          mean_type = 'arithmetic')
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=0.05, n_max = 1e10)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    print(data)


.. parsed-literal::

    Solution: 6.2519         
    AsianCall (Integrand Object)
    	volatility      [ 0.500  0.500  0.500]
    	start_price     [30 30 30]
    	strike_price    [25 25 25]
    	interest_rate   [ 0.010  0.010  0.010]
    	mean_type       ['arithmetic' 'arithmetic' 'arithmetic']
    	exercise_time   [ 1.000  1.000  1.000]
    IIDStdGaussian (Discrete Distribution Object)
    	mimics          StdGaussian
    BrownianMotion (True Measure Object)
    	dimension       [ 4 16 64]
    	time_vector     [array([ 0.250,  0.500,  0.750,  1.000])
    	                array([ 0.062,  0.125,  0.188, ...,  0.875,  0.938,  1.000])
    	                array([ 0.016,  0.031,  0.047, ...,  0.969,  0.984,  1.000])]
    CLT (Stopping Criterion Object)
    	abs_tol         0.050
    	rel_tol         0
    	n_max           10000000000
    	inflate         1.200
    	alpha           0.010
    MeanVarData (AccumData Object)
    	n               [ 278966.000  37778.000  7935.000]
    	n_total         327751
    	confid_int      [ 6.203  6.301]
    	time_total      0.112
    

