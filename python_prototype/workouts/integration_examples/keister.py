"""
Keister example
python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
"""

from qmcpy import *
from copy import deepcopy

bar = '\n'+'~'*100+'\n'

def keister(dimension=3, abs_tol=.1):
    print(bar)

    # CLT
    distribution = IIDStdUniform(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    stopping_criterion = CLT(integrand,abs_tol=abs_tol)
    solution,data = stopping_criterion.integrate()
    print('%s%s'%(data,bar))

    # CLTRep
    distribution = Lattice(dimension, scramble=True, replications=16, seed=7, backend='MPS')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CLTRep(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # MeanMC_g
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = MeanMC_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubLattice_g
    distribution = Lattice(dimension=dimension, scramble=True, replications=0, seed=7, backend='GAIL')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubLattice_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubSobol_g
    distribution = Sobol(dimension=dimension, scramble=True, replications=0, seed=7, backend='MPS')
    measure = Gaussian(distribution, covariance=1/2)
    integrand = Keister(measure)
    solution,data = CubSobol_g(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))


if __name__ == "__main__":
    keister()
