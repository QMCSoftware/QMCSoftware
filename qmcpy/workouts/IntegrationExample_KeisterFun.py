'''
Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin
3 Dimensonal Keister Function
    Run: python workouts/IntegrationExample_KeisterFun.py
    Save Output: python workouts/IntegrationExample_KeisterFun.py  > workouts/Outputs/ie_KeisterFun.txt
'''

from workouts import summary_qmc
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLTRep import CLTRep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrate import integrate
from algorithms.function.KeisterFun import KeisterFun
from algorithms.distribution import Measure

# IID std_gaussian
dim = 3
funObj = KeisterFun()
measureObj = Measure().iid_zmean_gaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(trueD=Measure().std_gaussian(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# IID std_uniform
dim = 3
funObj = KeisterFun()
measureObj = Measure().iid_zmean_gaussian(dimension=[dim], variance=[1 / 2])
distribObj = IIDDistribution(trueD=Measure().std_uniform(dimension=[dim]), rngSeed=7)
stopObj = CLTStopping(distribObj,absTol=1.5e-3)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Lattice
dim = 3
funObj = KeisterFun()
measureObj = Measure().iid_zmean_gaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(trueD=Measure().lattice(dimension=[dim]),rngSeed=7)
stopObj = CLTRep(distribObj,absTol=1.5e-3,nMax=1e6)
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom sobol
dim = 3
funObj = KeisterFun()
measureObj = Measure().iid_zmean_gaussian(dimension=[dim], variance=[1 / 2])
distribObj = QuasiRandom(trueD=Measure().sobol(dimension=[dim]), rngSeed=7)
stopObj = CLTRep(distribObj,absTol=0,nMax=1e6) # impossible tolerance so calculation is limited by sample budget
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
