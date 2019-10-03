"""
3 Dimensonal Keister Function
    Run: python workouts/IntegrationExample_KeisterFun.py
    Save Output: python workouts/IntegrationExample_KeisterFun.py  > outputs/ie_KeisterFun.txt
"""

from workouts import summary_qmc
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLTRep import CLTRep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrate import integrate
from algorithms.integrand.Keister import Keister
from algorithms.distribution.Measures import *

# IID std_uniform
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(true_distrib=StdUniform(dimension=[dim]), seed_rng=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# IID std_gaussian
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(true_distrib=StdGaussian(dimension=[dim]), seed_rng=7)
stopObj = CLTStopping(distribObj,abs_tol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Lattice
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(true_distrib=Lattice(dimension=[dim]),seed_rng=7)
stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom sobol
dim = 3
funObj = Keister()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(true_distrib=Sobol(dimension=[dim]), seed_rng=7)
stopObj = CLTRep(distribObj,abs_tol=.05,n_max=1e6) # impossible tolerance so calculation is limited by sample budget
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
