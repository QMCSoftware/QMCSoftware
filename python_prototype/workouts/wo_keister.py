#!/usr/bin/python_prototype/
"""
3 Dimensonal Keister Function
    Run: python_prototype workouts/wo_keister.py
    Save Output: python_prototype workouts/wo_keister.py  > outputs/ie_KeisterFun.txt
"""

from qmcpy.stop.clt_stopping import CLTStopping
from qmcpy.stop.clt_rep import CLTRep
from qmcpy.distribution.iid_distribution import IIDDistribution
from qmcpy.distribution.quasi_random import QuasiRandom
from qmcpy.integrate import integrate
from qmcpy.integrand.keister import Keister
from qmcpy.measures.measures import *
from qmcpy import print_summary

def test_distributions():
    # IID std_uniform
    dim = 3
    funObj = Keister()
    measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[dim]), seed_rng=7)
    stopObj = CLTStopping(distribObj,abs_tol=.05)
    sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,funObj,distribObj,dataObj)

    # IID std_gaussian
    dim = 3
    funObj = Keister()
    measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[dim]), seed_rng=7)
    stopObj = CLTStopping(distribObj,abs_tol=.05)
    sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,funObj,distribObj,dataObj)

    # QuasiRandom Lattice
    dim = 3
    funObj = Keister()
    measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribObj = QuasiRandom(true_distribution=Lattice(dimension=[dim]),seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6)
    sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,funObj,distribObj,dataObj)

    # QuasiRandom sobol
    dim = 3
    funObj = Keister()
    measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
    distribObj = QuasiRandom(true_distribution=Sobol(dimension=[dim]), seed_rng=7)
    stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6) # impossible tolerance so calculation is limited by sample budget
    sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
    print_summary(stopObj,measureObj,funObj,distribObj,dataObj)


test_distributions()