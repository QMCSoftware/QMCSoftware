#!/usr/bin/python_prototype/
"""
3 Dimensonal Keister Function
    Run: python workouts/wo_keister.py
    Save Output: python workouts/wo_keister.py  > outputs/ie_KeisterFun.txt
"""

from qmcpy import integrate
from qmcpy._util import summarize
from qmcpy.stop import CLT, CLTRep
from qmcpy.discrete_distribution import IIDDistribution, QuasiRandom
from qmcpy.integrand import Keister
from qmcpy.measures import *

def test_distributions_keister():
    # IID std_uniform
    dim = 3
    integrand = Keister()
    measure = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribution = IIDDistribution(true_distribution=StdUniform(dimension=[dim]),
                                 seed_rng=7)
    stop = CLT(distribution, abs_tol=.05)
    sol, data = integrate(integrand, measure, distribution, stop)
    data.summarize()

    # IID std_gaussian
    dim = 3
    integrand = Keister()
    measure = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribution = IIDDistribution(true_distribution=StdGaussian(dimension=[dim]),
                                 seed_rng=7)
    stop = CLT(distribution, abs_tol=.05)
    sol, data = integrate(integrand, measure, distribution, stop)
    data.summarize()

    # QuasiRandom Lattice
    dim = 3
    integrand = Keister()
    measure = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribution = QuasiRandom(true_distribution=Lattice(dimension=[dim]),
                             seed_rng=7)
    stop = CLTRep(distribution, abs_tol=.05, n_max=1e6)
    sol, data = integrate(integrand, measure, distribution, stop)
    data.summarize()

    # QuasiRandom sobol
    dim = 3
    integrand = Keister()
    measure = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribution = QuasiRandom(true_distribution=Sobol(dimension=[dim]),
                             seed_rng=7)
    stop = CLTRep(distribution, abs_tol=.05, n_max=1e6)
    # impossible tolerance so calculation is limited by sample budget
    sol, data = integrate(integrand, measure, distribution, stop)
    data.summarize()


test_distributions_keister()