"""
Single-Level and Multi-Level Asian Option Pricing Examples
    Run Workout:
        python workouts/wo_asian_option.py
    Save Output:
        python workouts/wo_asian_option.py  > outputs/examples/ie_AsianOption.txt
"""

from numpy import arange

from qmcpy import *


def test_distributions_asian_option(time_vec, dim, abs_tol):
    """
    Estimate Asian option value using various discrete sampling distributions.
    """

    # IID Standard Uniform ~ CLT
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    print(data)

    # IID Standard Uniform ~ CLT
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    print(data)

    if len(dim) == 1:  # CLTRep & MeanMC_g only implemented for single-level functions

        # IID Standard Uniform ~ MeanMC_g
        discrete_distrib = IIDStdUniform(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure, mean_type='geometric')
        stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        print(data)

        # IID Standard Uniform ~ MeanMC_g
        discrete_distrib = IIDStdGaussian(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure, interest_rate=.01, mean_type='geometric')
        stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        print(data)

        # Lattice ~ CLTRep
        discrete_distrib = Lattice(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure, volatility=.4, start_price=50, strike_price=40,
                              interest_rate=.02, mean_type='geometric')
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        print(data)

        # Sobol ~ CLTRep
        discrete_distrib = Sobol(rng_seed=7)
        true_measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(true_measure, volatility=.4, start_price=50, strike_price=40,
                              interest_rate=.02, mean_type='geometric')
        stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
        _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
        print(data)


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
Previo