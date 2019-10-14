#! /usr/bin/env python_prototype
"""
Single Level Asian Option Pricing
    Run: python workouts/wo_asian_option.py
    Save Output: python workouts/wo_asian_option.py  > outputs/ie_AsianOption.txt
"""

from numpy import arange
from qmcpy import integrate
from qmcpy.discrete_distribution import *
from qmcpy.integrand import AsianCall
from qmcpy.stop import CLT, CLTRep
from qmcpy.true_measure import BrownianMotion


def test_distributions_asian_option():
    """ Singl-Level Asian Option Pricing """
    time_vec = [arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]

    # IID Standard Uniform
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # IID Standard Uniform
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Lattice
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Sobol
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    """ Multi-Level Asian Option Pricing """
    # IID std_uniform
    time_vec = [
        arange(1 / 4, 5 / 4, 1 / 4),
        arange(1 / 16, 17 / 16, 1 / 16),
        arange(1 / 64, 65 / 64, 1 / 64)]
    dim = [len(tv) for tv in time_vec]

    # IID Standard Uniform
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # IID Standard Uniform
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Lattice
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Sobol
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = BrownianMotion(dim, time_vector=time_vec)
    integrand = AsianCall(true_measure)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.05)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()


test_distributions_asian_option()
