from qmcpy import *
from numpy import *

# Probably Brownian Motion or Asian Call problem
dimension = 16
tv = arange(1/dimension,1+1/dimension,1/dimension)
distribution = Lattice(dimension, seed=7) # taking 16*2^20 samples
#distribution = Sobol(dimension, seed=7) # taking 16*2^17 samples
measure = BrownianMotion(distribution,tv)
integrand = AsianCall(measure)
algorithm = CLTRep(integrand,abs_tol=.001)
solution,data = algorithm.integrate()
print(data)
