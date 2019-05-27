# Integrating a function using our community QMC framework

from numpy import arange

from CLTStopping import CLTStopping
from IIDDistribution import IIDDistribution
from integrate import integrate
from KeisterFun import KeisterFun 
from AsianCallFun import AsianCallFun
from measure import measure

''' An example with Keister's function integrated with respect to the uniform distribution over the unit cube '''
dim=3 # dimension for the Keister Example
measureObj = measure(measureName='IIDZMeanGaussian',dimension=dim,variance=1/2)
distribObj = IIDDistribution(trueD=measure(measureName='stdGaussian',dimension=dim)) # IID sampling
stopObj = CLTStopping() # stopping criterion for IID sampling using the Central Limit Theorem
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)

stopObj.absTol = 1e-3 # decrease tolerance
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)

stopObj.absTol = 0 # impossible tolerance
stopObj.nMax = 1e6 # calculation limited by sample budget
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)

''' A multilevel example of Asian option pricing '''
stopObj.absTol = 0.01 # increase tolerence
stopObj.nMax = 1e8 # pushing the sample budget back up
measureObj = measure(measureName='BrownianMotion',timeVector=arange(1/4,5/4,1/4))
OptionObj = AsianCallFun(measureObj) # 4 time steps
distribObj = IIDDistribution(trueD=measure(measureName='stdGaussian',dimension=4)) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)

measureObj = measure(measureName='BrownianMotion',timeVector=arange(1/64,65/64,1/64))
OptionObj = AsianCallFun(measureObj) # 64 time steps
distribObj = IIDDistribution(trueD=measure(measureName='stdGaussian',dimension=64)) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)

measureObj = measure(measureName='BrownianMotion',timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = IIDDistribution(trueD=measure(measureName='stdGaussian',dimension=[4,16,64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
print('sol =',sol,'\n',out)