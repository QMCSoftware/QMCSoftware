from qmcpy import *
from numpy import *

# Probably Brownian Motion or Asian Call problem
'''
distribution = Lattice(dimension=16, replications=16, scramble=True, seed=7, backend="GAIL")
#distribution = Sobol(dimension=16, replications=16, scramble=True, seed=7, backend="MPS")
#distribution = IIDStdGaussian(dimension=16, seed=7)
measure = BrownianMotion(distribution,time_vector=arange(1/16,17/16,1/16))
integrand = AsianCall(measure)
algorithm = CLTRep(integrand,abs_tol=.001)
solution,data = algorithm.integrate()
print(data)
'''

# CubLattice (parallel to matlab)
'''
distribution = Lattice(dimension=2, scramble=True, replications=0, seed=7, backend='GAIL')
measure = Uniform(distribution)
integrand = QuickConstruct(measure, lambda x: 5*x.sum(1))
algorithm = CubLattice_g(integrand, abs_tol=1e-4)
solution,data = algorithm.integrate()
print(solution)
'''