QMCPy Documentation
===================



Stopping Criterion Algorithms
-----------------------------

Abstract Stopping Criterion Class
.................................

.. automodule:: qmcpy.stopping_criterion._stopping_criterion
    :members:

Cubature Lattice Garunteed (qMC)
................................

.. automodule:: qmcpy.stopping_criterion.cub_lattice_g
    :members:

Cubature Sobol Garunteed (qMC)
..............................

.. automodule:: qmcpy.stopping_criterion.cub_sobol_g
    :members:

Multilevel Quasi-Monte Carlo (qMC)
..................................

.. automodule:: qmcpy.stopping_criterion.mlqmc
    :members:

Central Limit Theorem for Replications (qMC)
............................................

.. automodule:: qmcpy.stopping_criterion.clt_rep
    :members:

Multilevel Monte Carlo (MC)
...........................

.. automodule:: qmcpy.stopping_criterion.mlmc
    :members:

Mean Monte Carlo Garunteed (MC)
...............................

.. automodule:: qmcpy.stopping_criterion.mean_mc_g
    :members:

Central Limit Theorem (MC)
..........................

.. automodule:: qmcpy.stopping_criterion.clt
    :members:



Integrand Class
---------------

Abstract Integrand Class
........................

.. automodule:: qmcpy.integrand._integrand
    :members:

Keister Function
................

.. automodule:: qmcpy.integrand.keister
    :members:

Asian Call Option
.................

.. automodule:: qmcpy.integrand.asian_call
    :members:

Various Call Options by Milstein Discretization 
...............................................

.. automodule:: qmcpy.integrand.mlmc_call_options
    :members:

Custom Function
...............

.. automodule:: qmcpy.integrand.quick_construct
    :members:

Linear Function
...............

.. automodule:: qmcpy.integrand.linear
    :members:



Measure Class
-------------

Abstract Measure Class
......................

.. automodule:: qmcpy.true_measure._true_measure
    :members:

Uniform
.......

.. automodule:: qmcpy.true_measure.uniform
    :members:

Gaussian
........

.. automodule:: qmcpy.true_measure.gaussian
    :members:

Brownian Motion
...............

.. automodule:: qmcpy.true_measure.brownian_motion
    :members:

Lebesgue
........

.. automodule:: qmcpy.true_measure.lebesgue
    :members:

Identical to what Discrete Distribution Mimics
..................

.. automodule:: qmcpy.true_measure.identical_to_discrete
    :members:

Importance Sampling
...................

.. automodule:: qmcpy.true_measure.importance_sampling
    :members:



Discrete Distribution Class
---------------------------

Abstract Discrete Distribution Class
....................................

.. automodule:: qmcpy.discrete_distribution._discrete_distribution
    :members:

Lattice
.......

.. automodule:: qmcpy.discrete_distribution.lattice.lattice
    :members:

Sobol
.....

.. automodule:: qmcpy.discrete_distribution.sobol.sobol
    :members:

IID Standard Uniform
....................

.. automodule:: qmcpy.discrete_distribution.iid_std_uniform
    :members:

IID Standard Gaussian
.....................

.. automodule:: qmcpy.discrete_distribution.iid_std_gaussian
    :members:

Custom IID Distribution
.......................

.. automodule:: qmcpy.discrete_distribution.custom_iid_distribution
    :members:

Inverse CDF Sampling
....................

.. automodule:: qmcpy.discrete_distribution.inverse_cdf_sampling
    :members:

Acceptance Rejection Sampling
.............................

.. automodule:: qmcpy.discrete_distribution.acceptance_rejection_sampling
    :members:



Accumulate Data Class
---------------------

Abstract Accumulate Data Class
...................

.. automodule:: qmcpy.accumulate_data._accumulate_data
    :members:

Cubature Data (qMC)
...................

.. automodule:: qmcpy.accumulate_data.cubature_data
    :members:

Mean Variance for Replications Data (qMC)
.........................................

.. automodule:: qmcpy.accumulate_data.mean_var_data_rep
    :members:

Multilevel Data (MC)
....................

.. automodule:: qmcpy.accumulate_data.mlmc_data
    :members:
    
Mean Variance Data (MC)
.......................

.. automodule:: qmcpy.accumulate_data.mean_var_data
    :members:



Utilities
---------

Math Functions
..............

.. automodule:: qmcpy.util.math_functions
    :members:

Exceptions and Warnnings
.........................

.. automodule:: qmcpy.util.exceptions_warnings
    :members:

Abstraction Functions
.....................

.. automodule:: qmcpy.util.abstraction_functions
    :members:
