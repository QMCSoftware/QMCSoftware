''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''

from workouts import summary_qmc
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrate import integrate
from algorithms.function.KeisterFun import KeisterFun
from algorithms.distribution import measure

from numpy import arange,random
random.seed(7)

''' 3 Dimensonal Keister Function '''
# IID stdUniform
dim = 3
funObj = KeisterFun()
stopObj = CLTStopping()
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim]))
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# IID stdGaussian
dim = 3
funObj = KeisterFun()
stopObj = CLTStopping(absTol=1.5e-3)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdUniform(dimension=[dim]))
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Lattice
dim = 3
funObj = KeisterFun()
stopObj = CLT_Rep(absTol=1.5e-3,nMax=1e6)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = QuasiRandom(trueD=measure().lattice(dimension=[dim]))
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)

# QuasiRandom Sobol
dim = 3
funObj = KeisterFun()
stopObj = CLT_Rep(absTol=0,nMax=1e6) # impossible tolerance so calculation is limited by sample budget
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = QuasiRandom(trueD=measure().Sobol(dimension=[dim]))
sol,dataObj = integrate(funObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj)
