#! /usr/bin/env python_prototype
"""
Single Level Asian Option Pricing
    Run: python workouts/wo_asian_option.py
    Save Output: python workouts/wo_asian_option.py  > outputs/ie_AsianOption.txt
"""

from numpy import arange

from qmcpy import print_summary
from qmcpy.stop.clt_stopping import CLTStopping
from qmcpy.stop.clt_rep import CLTRep
from qmcpy.distribution.iid_distribution import IIDDistribution
from qmcpy.distribution.quasi_random import QuasiRandom
from qmcpy.integrate import integrate
from qmcpy.integrand.asian_call import AsianCall
from qmcpy.measures.measures import *

def test_distributions():
    # IID std_uniform
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[64]), seed_rng=7)
    stopObj = CLTStopping(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # IID std_gaussian
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[64]), seed_rng=7)
    stopObj = CLTStopping(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom lattice
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Lattice(dimension=[64]),seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom sobol
    measureObj = BrownianMotion(time_vector=[arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Sobol(dimension=[64]), seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    """ Multi-Level Asian Option Pricing """
    # IID std_uniform
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLTStopping(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # IID std_gaussian
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLTStopping(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom Lattice
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Lattice(dimension=[4,16,64]),seed_rng=7)
    stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

    # QuasiRandom sobol
    measureObj = BrownianMotion(time_vector=[arange(1 / 4, 5 / 4, 1 / 4), arange(1 / 16, 17 / 16, 1 / 16), arange(1 / 64, 65 / 64, 1 / 64)])
    OptionObj = AsianCall(measureObj)
    distribObj = QuasiRandom(true_distribution=Sobol(dimension=[4, 16, 64]), seed_rng=7)
    stopObj = CLTRep(distribObj,n_max=2**20,abs_tol=.05)
    sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,OptionObj,distribObj,dataObj)

test_distributions()
