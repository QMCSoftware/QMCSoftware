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
classes, ``Keister``, ``Gaussian``, ``Sobol`` and ``CLTRep``, as inputs
to the function ``integrate()``.

.. code:: ipython3

    dim = 3
    distribution = Lattice(dimension=dim, scramble=True, replications=16, seed=7, backend='MPS')
    measure = Gaussian(distribution, variance=1/2)
    integrand = Keister(measure)
    stopper = CLTRep(distribution,abs_tol=.05)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print(data)


.. parsed-literal::

    Solution: 2.1659         
    Keister (Integrand Object)
    Lattice (Discrete Distribution Object)
    	dimension       3
    	scramble        1
    	replications    16
    	seed            7
    	backend         mps
    	mimics          StdUniform
    Gaussian (True Measure Object)
    	distrib_name    Lattice
    	mean            0
    	variance        0.500
    CLTRep (Stopping Criterion Object)
    	inflate         1.200
    	alpha           0.010
    	abs_tol         0.050
    	rel_tol         0
    	n_init          256
    	n_max           1073741824
    MeanVarDataRep (Data Object)
    	replications    16
    	solution        2.166
    	sighat          0.011
    	n_total         256
    	confid_int      [ 2.164  2.168]
    	time_total      0.004
    


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

    time_vector = arange(1 / 64, 65 / 64, 1 / 64)
    distribution = Lattice(dimension=len(time_vector), scramble=True, replications=16, seed=7, backend='GAIL')
    measure = BrownianMotion(distribution, time_vector=time_vector)
    integrand = AsianCall(
        measure = measure,
        volatility = 0.5,
        start_price = 30,
        strike_price = 25,
        interest_rate = 0.01,
        mean_type = 'arithmetic')
    stopper = CLTRep(distribution, abs_tol=.05)
    solution,data = integrate(stopper, integrand, measure, distribution)
    print(data)


.. parsed-literal::

    Solution: 6.2588         
    AsianCall (Integrand Object)
    	volatility      0.500
    	start_price     30
    	strike_price    25
    	interest_rate   0.010
    	mean_type       arithmetic
    	_dim_frac       0
    Lattice (Discrete Distribution Object)
    	dimension       64
    	scramble        1
    	replications    16
    	seed            7
    	backend         gail
    	mimics          StdUniform
    BrownianMotion (True Measure Object)
    	distrib_name    Lattice
    	time_vector     [ 0.016  0.031  0.047 ...  0.969  0.984  1.000]
    CLTRep (Stopping Criterion Object)
    	inflate         1.200
    	alpha           0.010
    	abs_tol         0.050
    	rel_tol         0
    	n_init          256
    	n_max           1073741824
    MeanVarDataRep (Data Object)
    	replications    16
    	solution        6.259
    	sighat          0.021
    	n_total         4096
    	confid_int      [ 6.258  6.260]
    	time_total      0.552
    


European Arithmetic-Mean Asian Put Option: Multi-Level
------------------------------------------------------

This example is similar to the last one except that we use Gile's
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

    time_vector = [
        arange(1/4,5/4,1/4),
        arange(1/16,17/16,1/16),
        arange(1/64,65/64,1/64)]
    levels = len(time_vector)
    distributions = MultiLevelConstructor(levels,
        IIDStdGaussian,
            dimension = [len(tv) for tv in time_vector],
            seed = 7)
    measures = MultiLevelConstructor(levels,
        BrownianMotion,
            distribution = distributions,
            time_vector = time_vector)
    integrands = MultiLevelConstructor(levels,
        AsianCall,
            measure = measures,
            volatility = 0.5,
            start_price = 30,
            strike_price = 25,
            interest_rate = 0.01,
            mean_type = 'arithmetic')
    stopper = CLT(distributions, abs_tol=.05)
    solution,data = integrate(stopper, integrands, measures, distributions)
    print(data)


.. parsed-literal::

    Solution: 6.2389         
    MultiLevelConstructor (AsianCall Object)
    	volatility      [ 0.500  0.500  0.500]
    	start_price     [30 30 30]
    	strike_price    [25 25 25]
    	interest_rate   [ 0.010  0.010  0.010]
    	mean_type       ['arithmetic' 'arithmetic' 'arithmetic']
    	_dim_frac       [ 0.000  4.000  4.000]
    MultiLevelConstructor (IIDStdGaussian Object)
    	dimension       [ 4 16 64]
    	seed            [7 7 7]
    	mimics          ['StdGaussian' 'StdGaussian' 'StdGaussian']
    MultiLevelConstructor (BrownianMotion Object)
    	time_vector     [array([ 0.250,  0.500,  0.750,  1.000])
    	                array([ 0.062,  0.125,  0.188, ...,  0.875,  0.938,  1.000])
    	                array([ 0.016,  0.031,  0.047, ...,  0.969,  0.984,  1.000])]
    CLT (Stopping Criterion Object)
    	inflate         1.200
    	alpha           0.010
    	abs_tol         0.050
    	rel_tol         0
    	n_init          1024
    	n_max           10000000000
    MeanVarData (Data Object)
    	levels          3
    	solution        6.239
    	n               [284574  38751   6045]
    	n_total         332442
    	confid_int      [ 6.190  6.288]
    	time_total      0.106
    


