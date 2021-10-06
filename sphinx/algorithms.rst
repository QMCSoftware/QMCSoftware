QMCPy Documentation
===================

.. image:: uml/discrete_distribution_overview.png
.. image:: uml/true_measure_overview.png
.. image:: uml/integrand_overview.png
.. image:: uml/stopping_criterion_overview.png








Discrete Distribution Class
---------------------------

.. image:: uml/discrete_distribution_specific.png

Abstract Discrete Distribution Class
....................................

.. automodule:: qmcpy.discrete_distribution._discrete_distribution
    :members:

Digital Net Base 2
..................

.. automodule:: qmcpy.discrete_distribution.digital_net_b2.digital_net_b2
    :members:

Lattice
.......

.. automodule:: qmcpy.discrete_distribution.lattice.lattice
    :members:

Halton
......

.. automodule:: qmcpy.discrete_distribution.halton
    :members:

IID Standard Uniform
....................

.. automodule:: qmcpy.discrete_distribution.iid_std_uniform
    :members:







True Measure Class
------------------

.. image:: uml/true_measure_specific.png

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

Continuous Bernoulli
....................

.. automodule:: qmcpy.true_measure.bernoulli_cont
    :members:

Johnson's SU
............

.. automodule:: qmcpy.true_measure.johnsons_su
    :members:

Kumaraswamy
...........

.. automodule:: qmcpy.true_measure.kumaraswamy
    :members:

SciPy Wrapper
.............

.. automodule:: qmcpy.true_measure.scipy_wrapper
    :members:






Integrand Class
---------------

.. image:: uml/integrand_specific.png

Abstract Integrand Class
........................

.. automodule:: qmcpy.integrand._integrand
    :members:

Custom Function
...............

.. automodule:: qmcpy.integrand.custom_fun
    :members:

Keister Function
................

.. automodule:: qmcpy.integrand.keister
    :members:

Box Integral
.............

.. automodule:: qmcpy.integrand.box_integral
    :members:

European Option
...............

.. automodule:: qmcpy.integrand.european_option
    :members:

Asian Option
.................

.. automodule:: qmcpy.integrand.asian_option
    :members:

Multilevel Call Options with Milstein Discretization
.....................................................

.. automodule:: qmcpy.integrand.ml_call_options
    :members:

Linear Function
...............

.. automodule:: qmcpy.integrand.linear0
    :members:

Sobol' Indices
..............

.. automodule:: qmcpy.integrand.sobol_indices
    :members:






Stopping Criterion Algorithms
-----------------------------

.. image:: uml/stopping_criterion_specific.png

Abstract Stopping Criterion Class
.................................

.. automodule:: qmcpy.stopping_criterion._stopping_criterion
    :members:

Guaranteed Digital Net Cubature (QMC)
.....................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_net_g
    :members:

Guaranteed Lattice Cubature (QMC)
.................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_lattice_g
    :members:

Bayesian Lattice Cubature (QMC)
................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_bayes_lattice_g
    :members:

Bayesian Digital Net Cubature (QMC)
...................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_bayes_net_g
    :members:

CLT QMC Cubature (with Replications)
....................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_clt
    :members:

Guaranteed MC Cubature
......................

.. automodule:: qmcpy.stopping_criterion.cub_mc_g
    :members:

CLT MC Cubature
..........................

.. automodule:: qmcpy.stopping_criterion.cub_mc_clt
    :members:

Continuation Multilevel QMC Cubature
....................................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_ml_cont
    :members:

Multilevel QMC Cubature
.......................

.. automodule:: qmcpy.stopping_criterion.cub_qmc_ml
    :members:

Continuation Multilevel MC Cubature
...................................

.. automodule:: qmcpy.stopping_criterion.cub_mc_ml_cont
    :members:

Multilevel MC Cubature
......................

.. automodule:: qmcpy.stopping_criterion.cub_mc_ml
    :members:






Utilities
---------

.. image:: uml/util_err.png
.. image:: uml/util_warn.png

.. automodule:: qmcpy.util.latnetbuilder_linker
    :members: