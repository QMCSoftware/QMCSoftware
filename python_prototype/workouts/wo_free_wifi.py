"""
WARNING: This script takes 
Mathematica (Wolfram Alpha): integral_(-2)^2 sqrt(4 - x^2) (1/2 + x^3 cos(x/2)) dx = 3.14159
Answer looks to be pi from both Mathematica and out QMCPy software
"""

from qmcpy import *

from numpy import *
from time import time

t0 = time()
integrand = QuickConstruct(
        dimension = 1,
        custom_fun = lambda x: sqrt(4-x**2) * (1/2 + x**3 * cos(x/2)))
true_measure = Lebesgue(dimension=1, lower_bound=-2, upper_bound=2)
discrete_distrib = Lattice(rng_seed=7)
stop_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=2.5e-10, n_max=2**30)
sol,data_obj = integrate(integrand, true_measure, discrete_distrib, stop_criterion)
password = str(sol).replace('.','')[:10]
print("Password:",password) # 3141592653
print('Time: %.2f sec'%(time()-t0)) # around 30 seconds