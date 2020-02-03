"""
Single-Level and Multi-Level Asian Option Pricing Examples
    Save Output:
        python workouts/wo_integration_examples/wo_asian_option.py  > outputs/wo_integration_examples/ie_AsianOption.txt
"""

from numpy import arange
from qmcpy import *

volatility = .5
start_price = 30
strike_price = 25
interest_rate = .01
mean_type = 'geometric'


def test_distributions_asian_option(time_vec, dim, abs_tol):
    """
    Estimate Asian option value using various discrete sampling distributions.
    """

    # IID Standard Uniform ~ CLT
    distrib = IIDStdUniform(rng_seed=7)
    measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(measure,
                          volatility=volatility,
                          start_price=start_price,
                          strike_price=strike_price,
                          interest_rate=interest_rate,
                          mean_type=mean_type)
    stopping_criterion = CLT(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # IID Standard Gaussian ~ CLT
    distrib = IIDStdGaussian(rng_seed=7)
    measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(measure,
                          volatility=volatility,
                          start_price=start_price,
                          strike_price=strike_price,
                          interest_rate=interest_rate,
                          mean_type=mean_type)
    stopping_criterion = CLT(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    if len(dim) == 1:  # CLTRep & MeanMC_g only implemented for single-level functions

        # IID Standard Uniform ~ MeanMC_g
        distrib = IIDStdUniform(rng_seed=7)
        measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(measure,
                              volatility=volatility,
                              start_price=start_price,
                              strike_price=strike_price,
                              interest_rate=interest_rate,
                              mean_type=mean_type)
        stopping_criterion = MeanMC_g(distrib, measure,
                                      abs_tol=abs_tol)
        _, data = integrate(integrand, measure,
                            distrib, stopping_criterion)
        print(data)

        # IID Standard Gaussian ~ MeanMC_g
        distrib = IIDStdGaussian(rng_seed=7)
        measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(measure,
                              volatility=volatility,
                              start_price=start_price,
                              strike_price=strike_price,
                              interest_rate=interest_rate,
                              mean_type=mean_type)
        stopping_criterion = MeanMC_g(distrib, measure,
                                      abs_tol=abs_tol)
        _, data = integrate(integrand, measure,
                            distrib, stopping_criterion)
        print(data)

        # Lattice ~ CLTRep
        distrib = Lattice(rng_seed=7)
        measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(measure,
                              volatility=volatility,
                              start_price=start_price,
                              strike_price=strike_price,
                              interest_rate=interest_rate,
                              mean_type=mean_type)
        stopping_criterion = CLTRep(distrib, measure,
                                    abs_tol=abs_tol)
        _, data = integrate(integrand, measure,
                            distrib, stopping_criterion)
        print(data)

        # Sobol ~ CLTRep
        distrib = Sobol(rng_seed=7)
        measure = BrownianMotion(dim, time_vector=time_vec)
        integrand = AsianCall(measure,
                              volatility=volatility,
                              start_price=start_price,
                              strike_price=strike_price,
                              interest_rate=interest_rate,
                              mean_type=mean_type)
        stopping_criterion = CLTRep(distrib, measure,
                                    abs_tol=abs_tol)
        _, data = integrate(integrand, measure,
                            distrib, stopping_criterion)
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
