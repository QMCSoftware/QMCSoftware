from qmcpy import *
from numpy import *

# Probably Brownian Motion or Asian Call problem

dimension = 16
tv = arange(1/dimension,1+1/dimension,1/dimension)
distribution = Lattice(dimension, replications=16, seed=7)
#distribution = Sobol(dimension, replications=16, seed=7)
measure = BrownianMotion(distribution,tv)
integrand = AsianCall(measure)
algorithm = CLTRep(integrand,abs_tol=.001)
solution,data = algorithm.integrate()
print(data)

# CubLattice_g (parallel to matlab) (stilde error. Same with CubSobol_g)
'''
distribution = Lattice(dimension=2, scramble=True, replications=0, seed=7, backend='GAIL')
measure = Uniform(distribution)
integrand = QuickConstruct(measure, lambda x: 5*x.sum(1))
algorithm = CubLattice_g(integrand, abs_tol=1e-4, check_cone=True)
solution,data = algorithm.integrate()
print(solution)
'''

# CubSobol_g (parallel to matlab) (stilde error. Same with CubSobol_g)
'''
distribution = Sobol(dimension=2, scramble=True, replications=0, seed=7, backend='GAIL')
measure = Uniform(distribution)
integrand = QuickConstruct(measure, lambda x: 5*x.sum(1))
algorithm = CubSobol_g(integrand, abs_tol=1e-4, check_cone=True)
solution,data = algorithm.integrate()
print(solution)
'''