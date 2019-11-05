"""
Single-Level and Multi-Level Asian Option Pricing Examples
    Run Workout: python workouts/wo_asian_option_n_max.py
    Save Output: python workouts/wo_asian_option_n_max.py > outputs/examples/ie_AsianOption_n_max.txt
"""

from numpy import arange
from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdGaussian, IIDStdUniform, Lattice, Sobol
from qmcpy.integrand import AsianCall
from qmcpy.stopping_criterion import CLT, CLTRep, MeanMC_g
from qmcpy.true_measure import BrownianMotion


def test_distributions_asian_option(time_vec, dim, abs_tol):
    """
    Estimate Asian option value using various discrete sampling distributions
    with the restriction to use less than maximum number of sample points.
    """
    
    # IID Standard Uniform ~ CLT
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # IID Standard Uniform ~ CLT
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    if len(dim) == 1: # CLTRep & MeanMC_g only implemented for single-level functions
        
        # IID Standard Uniform ~ MeanMC_g
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        data.summarize()

        # IID Standard Uniform ~ MeanMC_g
        discrete_distrib = IIDStdGaussian(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        data.summarize()

        # Lattice ~ CLTRep
        discrete_distrib = Lattice(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol,
                                    n_init=64, n_max=500)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        data.summarize()

        # Sobol ~ CLTRep
        discrete_distrib = Sobol(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure)
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol,
                                    n_init=64, n_max=500)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        data.summarize()


if __name__ == "__main__":
    # Singl-Level Asian Option Pricing
    TIME_VEC1 = [arange(1 / 64, 65 / 64, 1 / 64)]
    DIM1 = [len(tv) for tv in TIME_VEC1]
    test_distributions_asian_option(TIME_VEC1, DIM1, abs_tol=.05)

    # Multi-Level Asian Option Pricing
    TIME_VEC2 = [arange(1 / 4, 5 / 4, 1 / 4),
                 arange(1 / 16, 17 / 16, 1 / 16),
                 arange(1 / 64, 65 / 64, 1 / 64)]
    DIM2 = [len(tv) for tv in TIME_VEC2]
    test_distributions_asian_option(TIME_VEC2, DIM2, abs_tol=.05)
