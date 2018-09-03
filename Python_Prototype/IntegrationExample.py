# Originally developed by Doctor Fred Hickernell in MATLAB
# Translation by Sou-Cheng Choi and Aleksei Sorokin

'''
How to integrate a function using our community QMC framework
An example with KeisterFun's function integrated with respect to the uniform
distribution over the unit cube
'''
from CLTStopping import CLTStopping as CLTStopping
from IIDDistribution import IIDDistribution as IIDDistribution
from integrate import integrate as integrate
from KeisterFun import KeisterFun as KeisterFun

stopObj = CLTStopping() #stopping criterion for IID sampling using the Central Limit Theorem
distribObj = IIDDistribution() #IID sampling with uniform distribution
funObj = KeisterFun()
sol, out = integrate(funObj, distribObj, stopObj)
stopObj.absTol = 1e-3  # decrease tolerance
sol, out = integrate(funObj, distribObj, stopObj)
stopObj.absTol = 0  # impossible tolerance
stopObj.nMax = 1e6  # calculation limited by sample budget
sol, out = integrate(funObj, distribObj, stopObj)


# A multilevel example of Asian option pricing

distribObj.trueDistribution = 'normal' # Change to normal distribution
stopObj.absTol = 0.01 # increase tolerance
stopObj.nMax = 1e8; # pushing the sample budget back up
OptionObj = AsianCallFun(4) # 4 time steps
sol,out = integrate(OptionObj, distribObj, stopObj)
"""
OptionObj = AsianCallFun(64) # single level, 64 time steps
sol,out = integrate(OptionObj, distribObj, stopObj)

OptionObj = AsianCallFun([4 4 4]) # multilevel, 64 time steps, faster
sol,out = integrate(OptionObj, distribObj, stopObj)

"""

