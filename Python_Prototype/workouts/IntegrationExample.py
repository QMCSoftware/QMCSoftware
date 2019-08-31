''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
# Integrating a function using our community QMC framework

from numpy import arange,random
random.seed(7)

from algorithms.CLTStopping import CLTStopping
from algorithms.CLT_Rep import CLT_Rep
from algorithms.IIDDistribution import IIDDistribution
from algorithms.Mesh import Mesh
from algorithms.integrate import integrate
from algorithms.KeisterFun import KeisterFun
from algorithms.AsianCallFun import AsianCallFun
from algorithms.measure import measure

f = open('Outputs/ie_python.txt','w')
def output(sol,dataObj):
    s = 'sol = %.4f\n%s'%(sol,dataObj)
    print('\n'+s)
    f.write(s+'\n\n')

''' An example with Keister's function integrated with respect to the uniform distribution over the unit cube '''
dim = 3 # dimension for the Keister Example
stopObj = CLTStopping() # stopping criterion for IID sampling using the Central Limit Theorem
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim])) # IID sampling
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

# Same example as above but with decreased tolerance
dim = 3
stopObj = CLTStopping(absTol=1e-3)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim]))
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

# Same example as above but with impossible tolerance and calculation limited by sample budget 
dim = 3
stopObj = CLTStopping(absTol=0,nMax=1e6) # impossible tolerance so calculation is limited by sample budget
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[dim]))
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj)
output(sol,out)

''' A multilevel example of Asian option pricing '''
stopObj = CLTStopping()
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4)])
OptionObj = AsianCallFun(measureObj) # 4 time steps
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[4])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

stopObj = CLTStopping(absTol=.05)
measureObj = measure().BrownianMotion(timeVector=[arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # 64 time steps
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

stopObj = CLTStopping(absTol=.01)
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[4,16,64])) # IID sampling
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

''' Single and multilevel example of CLT_Rep (stopping_criterion) paired with meanvarData_Rep (accumData) '''
dim = 3
stopObj = CLT_Rep(nInit=4,nMax=2**15,absTol=.01)
measureObj = measure().IIDZMeanGaussian(dimension=[dim],variance=[1/2])
distribObj = Mesh(trueD=measure().mesh(dimension=[dim],meshType='lattice')) # Uniform Sampling from latticeseq_b2.py 
sol,out = integrate(KeisterFun(),measureObj,distribObj,stopObj) # KeisterFun()
output(sol,out)

# Lattice Mesh
stopObj = CLT_Rep(nMax=2**20,absTol=.01)
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = Mesh(trueD=measure().mesh(dimension=[4,16,64],meshType='lattice')) # Uniform Sampling from latticeseq_b2.py 
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

# Sobol Mesh
stopObj = CLT_Rep(nMax=2**20,absTol=.01)
measureObj = measure().BrownianMotion(timeVector=[arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)])
OptionObj = AsianCallFun(measureObj) # multi-level
distribObj = Mesh(trueD=measure().mesh(dimension=[4,16,64],meshType='sobol')) # Uniform Sampling from latticeseq_b2.py 
sol,out = integrate(OptionObj,measureObj,distribObj,stopObj)
output(sol,out)

f.close()