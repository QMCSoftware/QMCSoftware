"""
3 Dimensonal Keister Function
    Run: python workouts/IntegrationExample_KeisterFun.py
    Save Output: python workouts/IntegrationExample_KeisterFun.py  > outputs/ie_KeisterFun.txt
"""

from workouts import summary_qmc
from algorithms.stop.clt_stopping import CLTStopping
from algorithms.stop.clt_rep import CLTRep
from algorithms.distribution.iid_distribution import IIDDistribution
from algorithms.distribution.quasi_random import QuasiRandom
from algorithms.integrate import integrate
from algorithms.integrand.keister import Keister
from algorithms.measures.measures import *

# IID std_uniform
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(true_distribution=StdUniform(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# IID std_gaussian
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(true_distribution=StdGaussian(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Lattice
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(true_distribution=Lattice(dimension=[dim]),rngSeed=7)
stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom sobol
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(true_distribution=Sobol(dimension=[dim]), rngSeed=7)
stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6) # impossible tolerance so calculation is limited by sample budget
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
