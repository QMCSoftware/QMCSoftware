# Integrating a function using our community QMC framework

from numpy import arange

from CLTStopping import CLTStopping as CLTStopping
from IIDDistribution import IIDDistribution as IIDDistribution
from integrate import integrate as integrate
from KeisterFun import KeisterFun as KeisterFun
from AsianCallFun import AsianCallFun
from util import new_qmc_problem

''' An example with Keister's function integrated with respect to the uniform distribution over the unit cube '''
new_qmc_problem() # empties class lists
dim=3 # dimension for the Keister Example
measureObj = IIDZMeanGaussian(measure,dimension=dim,variance=1/2)
distribObj = IIDDistribution(trueD=stdGaussian(measure,dimension=dim)) # IID sampling
stopObj = CLTStopping() # stopping criterion for IID sampling using the Central Limit Theorem
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
stopObj.absTol = 1e-3 # decrease tolerance
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
stopObj.absTol = 0 # impossible tolerance
stopObj.nMax = 1e6 # calculation limited by sample budget
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)

''' A multilevel example of Asian option pricing '''
new_qmc_problem() # empties class lists
stopObj.absTol = 0.01 # increase tolerence
stopObj.nMax = 1e8 # pushing the sample budget back up
measureObj = BrownianMotion(measure,timeVector=arange(1/4,5/4,1/4))
OptionObj = AsianCallFun(measureObj) # 4 time steps
distribObj = IIDDistribution(trueD=stdGaussian(measure,dimension=4)) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)

measureObj = BrownianMotion(measure,timeVector=arange(1/64,65/64,1/64))
OptionObj = AsianCallFun(measureObj) # 64 time steps
distribObj = IIDDistribution(trueD=stdGaussian(measure,dimension=64)) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)

measureObj = BrownianMotion(measure,timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = IIDDistribution(trueD=stdGaussian(measure,dimension=[4,16,64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)





