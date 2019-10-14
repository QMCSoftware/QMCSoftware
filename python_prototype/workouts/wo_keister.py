#!/usr/bin/python_prototype/
"""
3 Dimensonal Keister Function
    Run: python workouts/wo_keister.py
    Save Output: python workouts/wo_keister.py  > outputs/ie_KeisterFun.txt
"""

from qmcpy import integrate
from qmcpy.discrete_distribution import *
from qmcpy.integrand import Keister
from qmcpy.stop import CLT, CLTRep
from qmcpy.true_measure import Gaussian


def test_distributions_keister():
    dim = 3

    # IID Standard Uniform
    integrand = Keister()
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # IID Standard Gaussian
    integrand = Keister()
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLT(discrete_distrib, true_measure, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Lattice
    integrand = Keister()
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Sobol
    integrand = Keister()
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=.01)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()


test_distributions_keister()
