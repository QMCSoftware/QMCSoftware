# Mathematica: 
#   N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]

from qmcpy import *
from copy import deepcopy

d = 2
true_value = 1.808186429263620
abs_tol = .0005
rel_tol = 0
bar = '~'*100+'\n'
print(bar)

# CLT
distribution = IIDStdUniform(dimension=d, seed=7)
measure = Gaussian(distribution, variance=1/2)
integrand = Keister(measure)
stopper = CLT(distribution,abs_tol=abs_tol,rel_tol=rel_tol)
solution,data = integrate(stopper,integrand,measure,distribution)
print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))


# CLTRep
distribution = Lattice(dimension=d, scramble=True, replications=16, seed=7, backend='MPS')
measure = Gaussian(distribution, variance=1/2)
integrand = Keister(measure)
stopper = CLTRep(distribution,abs_tol=abs_tol,rel_tol=rel_tol)
solution,data = integrate(stopper,integrand,measure,distribution)
print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

# MeanMC_g
distribution = IIDStdGaussian(dimension=d, seed=7)
measure = Gaussian(distribution, variance=1/2)
integrand = Keister(measure)
stopper = MeanMC_g(distribution,abs_tol=abs_tol,rel_tol=rel_tol)
solution,data = integrate(stopper,integrand,measure,distribution)
print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

# CubLattice
distribution = Lattice(dimension=d, scramble=True, replications=0, seed=7, backend='GAIL')
measure = Gaussian(distribution, variance=1/2)
integrand = Keister(measure)
stopper = CubLattice_g(distribution,abs_tol=abs_tol,rel_tol=rel_tol)
solution,data = integrate(stopper,integrand,measure,distribution)
print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

# CLT Multi-Level


# CubSobol