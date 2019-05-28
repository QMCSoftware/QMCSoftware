# Integrating a function using our community QMC framework

from numpy import arange
import sys
# Supress ALL warning from RandomState package
import warnings
warnings.simplefilter("ignore")#,category=RandomStateDeprecationWarning)

from CLTStopping import CLTStopping
from IIDDistribution import IIDDistribution
from integrate import integrate
from KeisterFun import KeisterFun 
from AsianCallFun import AsianCallFun
from measure import measure

f = open('Outputs/ie_python.txt','w')
def output(sol,dataObj):
    s = 'sol = %.4f\n%s'%(sol,dataObj)
    print('\n'+s)
    f.write(s+'\n\n')

'''An example with Keister's function integrated with respect to the uniform distribution over the unit cube'''
dim = 3 # dimension for the Keister Example
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim])) # IID sampling
stopObj = CLTStopping() # stopping criterion for IID sampling using the Central Limit Theorem
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

stopObj.absTol = 1e-3 # decrease tolerance
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

stopObj.absTol = 0 # impossible tolerance
stopObj.nMax = 1e6 # calculation limited by sample budget
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

''' A multilevel example of Asian option pricing '''
stopObj.absTol = 0.01 # increase tolerence
stopObj.nMax = 1e8 # pushing the sample budget back up
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4)])
OptionObj = AsianCallFun(measureObj) # 4 time steps
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[4])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # 64 time steps
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[4,16,64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

f.close()