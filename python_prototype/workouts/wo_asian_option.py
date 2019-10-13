#! /usr/bin/env python_prototype
"""
Single Level Asian Option Pricing
    Run: python workouts/wo_asian_option.py
    Save Output: python workouts/wo_asian_option.py  > outputs/ie_AsianOption.txt
"""

from numpy import arange

from qmcpy import integrate
from qmcpy._util import summarize
from qmcpy.stop import CLT, CLTRep
from qmcpy.discrete_distribution import IIDDistribution, QuasiRandom
from qmcpy.integrand import AsianCall
from qmcpy.measures import *

def test_distributions_asian_option():
    # IID std_uniform
    measure = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    option = AsianCall(measure)
    distribution = IIDDistribution(true_distribution=StdUniform(dimension=[64]),
                                   seed_rng=7)
    stop = CLT(distribution, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # IID std_gaussian
    measure = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    option = AsianCall(measure)
    distribution = IIDDistribution(true_distribution=StdGaussian(dimension=[64]), seed_rng=7)
    stop = CLT(distribution, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # QuasiRandom lattice
    measure = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    option = AsianCall(measure)
    distribution = QuasiRandom(true_distribution=Lattice(dimension=[64]),
                               seed_rng=7)
    stop = CLTRep(distribution, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # QuasiRandom sobol
    measure = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    option = AsianCall(measure)
    distribution = QuasiRandom(true_distribution=Sobol(dimension=[64]),
                               seed_rng=7)
    stop = CLTRep(distribution, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    """ Multi-Level Asian Option Pricing """
    # IID std_uniform
    time_vector = [arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16),
                   arange(1 / 64, 65 / 64, 1 / 64)]
    measure = BrownianMotion(time_vector=time_vector)
    option = AsianCall(measure)
    distribution = IIDDistribution(
        true_distribution=StdUniform(dimension=[4, 16, 64]), seed_rng=7)
    stop = CLT(distribution, n_max=2 ** 20, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # IID std_gaussian
    measure = BrownianMotion(time_vector=time_vector)
    option = AsianCall(measure)
    distribution = IIDDistribution(
        true_distribution=StdGaussian(dimension=[4, 16, 64]), seed_rng=7)
    stop = CLT(distribution, n_max=2 ** 20, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # QuasiRandom Lattice
    measure = BrownianMotion(time_vector=time_vector)
    option = AsianCall(measure)
    distribution = QuasiRandom(true_distribution=Lattice(dimension=[4, 16, 64]),
                               seed_rng=7)
    stop = CLTRep(distribution, n_max=2 ** 20, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

    # QuasiRandom sobol
    measure = BrownianMotion(time_vector=time_vector)
    option = AsianCall(measure)
    distribution = QuasiRandom(true_distribution=Sobol(dimension=[4, 16, 64]),
                               seed_rng=7)
    stop = CLTRep(distribution, n_max=2 ** 20, abs_tol=.05)
    sol, data = integrate(option, measure, distribution, stop)
    data.summarize()

test_distributions_asian_option()
