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
from algorithms.integrand.KeisterFun import KeisterFun
from algorithms.distribution.Measures import *

# IID std_uniform
dim = 3
funObj = KeisterFun()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(trueD=StdUniform(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj,absTol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# IID std_gaussian
dim = 3
funObj = KeisterFun()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(trueD=StdGaussian(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj,absTol=.05)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Lattice
dim = 3
funObj = KeisterFun()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(trueD=Lattice(dimension=[dim]),rngSeed=7)
stopObj = CLTRep(distribObj,absTol=.05,nMax=1e6)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom sobol
dim = 3
funObj = KeisterFun()
measureObj = IIDZeroMeanGaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(trueD=Sobol(dimension=[dim]), rngSeed=7)
stopObj = CLTRep(distribObj,absTol=.05,nMax=1e6) # impossible tolerance so calculation is limited by sample budget
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
