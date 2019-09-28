'''
Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin
Single Level Asian Option Pricing
    Run: python workouts/IntegrationExample_AsianOption.py
    Save Output: python workouts/IntegrationExample_AsianOption.py  > workouts/Outputs/ie_AsianOption.txt
'''

from numpy import arange

from workouts import summary_qmc
from algorithms.stop.CLTStopping import CLTStopping
from algorithms.stop.CLT_Rep import CLT_Rep
from algorithms.distribution.IIDDistribution import IIDDistribution
from algorithms.distribution.QuasiRandom import QuasiRandom
from algorithms.integrate import integrate
from algorithms.function.AsianCallFun import AsianCallFun
from algorithms.distribution import measure

# IID stdUniform
measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = IIDDistribution(trueD=measure().stdUniform(dimension=[64]),rngSeed=7)
stopObj = CLTStopping(distribObj,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# IID stdGaussian
measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[64]),rngSeed=7)
stopObj = CLTStopping(distribObj,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom lattice
measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = QuasiRandom(trueD=measure().lattice(dimension=[64]),rngSeed=7)
stopObj = CLT_Rep(distribObj,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom Sobol
measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = QuasiRandom(trueD=measure().Sobol(dimension=[64]),rngSeed=7)
stopObj = CLT_Rep(distribObj,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

''' Multi-Level Asian Option Pricing '''
# IID stdUniform
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = IIDDistribution(trueD=measure().stdUniform(dimension=[4,16,64]),rngSeed=7)
stopObj = CLTStopping(distribObj,nMax=2**20,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# IID stdGaussian
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[4,16,64]),rngSeed=7)
stopObj = CLTStopping(distribObj,nMax=2**20,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom Lattice
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = QuasiRandom(trueD=measure().lattice(dimension=[4,16,64]),rngSeed=7)
stopObj = CLT_Rep(distribObj,nMax=2**20,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)

# QuasiRandom Sobol
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj)
distribObj = QuasiRandom(trueD=measure().Sobol(dimension=[4,16,64]),rngSeed=7)
stopObj = CLT_Rep(distribObj,nMax=2**20,absTol=.05)
sol,dataObj = integrate(OptionObj,measureObj,distribObj,stopObj)
summary_qmc(stopObj,measureObj,OptionObj,distribObj,dataObj)
