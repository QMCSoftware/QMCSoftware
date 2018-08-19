# Originally developed by Doctor Fred Hickernell in MATLAB
# Translation by Sou-Cheng Choi and Aleksei Sorokin

'''
How to integrate a function using our community QMC framework
An example with Keister's function integrated with respect to the uniform
distribution over the unit cube
'''

from QMC import QMC

test_qmc = QMC('meanVar','IID','Keister','CLT')
sol, out = test_qmc.integrate()
print(sol)

test_qmc = QMC('meanVar','IID','Keister','CLT')
test_qmc.stp.absTol = 1e-3  # decrease tolerance
sol, out = test_qmc.integrate()
print(sol)

test_qmc = QMC('meanVar','IID','Keister','CLT')
test_qmc.stp.absTol = 0  # impossible tolerance
test_qmc.stp.nMax = 1e6  # calculation limited by sample budget
sol, out = test_qmc.integrate()
print(sol)

# A multilevel example of Asian option pricing
"""
distribObj.trueDistribution = 'normal' # Change to normal distribution
stopObj.absTol = 0.01 # increase tolerance
stopObj.nMax = 1e8; # pushing the sample budget back up
OptionObj = AsianCallFun(4) # 4 time steps
sol,out = integrate(OptionObj, distribObj, stopObj)

OptionObj = AsianCallFun(64) # single level, 64 time steps
sol,out = integrate(OptionObj, distribObj, stopObj)

OptionObj = AsianCallFun([4 4 4]) # multilevel, 64 time steps, faster
sol,out = integrate(OptionObj, distribObj, stopObj)
"""
