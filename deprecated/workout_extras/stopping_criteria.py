"""
Sample DiscreteDistribution objects and usage
Mathematica: N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
python workouts/example_constructions/stopping_criteria.py > outputs/example_constructions/stopping_criteria.log
"""

from qmcpy import *
from copy import deepcopy


def stopping_criteria():
    d = 2
    true_value = 1.808186429263620
    abs_tol = .005
    rel_tol = 0
    bar = '\n'+'~'*100+'\n'
    print(bar)

    # CLT
    distribution = IIDStdUniform(dimension=d, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLT(integrand,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
    print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

    # CLTRep
    distribution = Lattice(dimension=d, scramble=True, seed=7, backend='MPS')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLTRep(integrand,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
    print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

    # MeanMC_g
    distribution = IIDStdGaussian(dimension=d, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = MeanMC_g(integrand,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
    print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

    # CubLattice_g
    distribution = Lattice(dimension=d, scramble=True, seed=7, backend='GAIL')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubLattice_g(integrand,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
    print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))

    # CubSobol_g
    '''
    distribution = Sobol(dimension=d, scramble=True, seed=7, backend='QRNG')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubSobol_g(integrand,abs_tol=abs_tol,rel_tol=rel_tol).integrate()
    print('%s\nMeets tolerance: %s\n%s'%(data,abs(solution-true_value)<abs_tol,bar))
    '''
    
if __name__ == '__main__':
    stopping_criteria()