"""
Single-Level and Multi-Level Asian Option Pricing Examples
    Run Workout:    python workouts/wo_asian_option_n_max.py
    Save Output:    python workouts/wo_asian_option_n_max.py  > outputs/ie_AsianOption_n_max.txt
"""

from numpy import arange
from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdUniform, IIDStdGaussian, Lattice, Sobol
from qmcpy.integrand import AsianCall
from qmcpy.stopping_criterion import CLT, CLTRep
from qmcpy.true_measure import BrownianMotion


def test_distributions_asian_option(time_vec, dim, abs_tol):

    # IID Standard Uniform
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
    sol, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # IID Standard Uniform
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=5000)
    sol, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # Lattice
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=500)
    sol, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # Sobol
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol, n_init=64, n_max=500)
    sol, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()


if __name__ == "__main__":
    # Singl-Level Asian Option Pricing
    time_vec = [arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]
    test_distributions_asian_option(time_vec, dim, abs_tol=.05)

    # Multi-Level Asian Option Pricing
    time_vec = [
        arange(1 / 4, 5 / 4, 1 / 4),
        arange(1 / 16, 17 / 16, 1 / 16),
        arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]
    test_distributions_asian_option(time_vec, dim, abs_tol=.05)
