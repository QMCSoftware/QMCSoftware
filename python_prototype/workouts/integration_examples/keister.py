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
    measure = Gaussian(distribution, variance=1/2)
    integrand = Keister(measure)
    stopper = CLT(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # CLTRep
    distribution = Lattice(dimension, scramble=True, replications=16, seed=7, backend='MPS')
    measure = Gaussian(distribution, variance=1/2)
    integrand = Keister(measure)
    stopper = CLTRep(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # MeanMC_g
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = Gaussian(distribution, variance=1/2)
    integrand = Keister(measure)
    stopper = MeanMC_g(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))

    # CubLattice
    distribution = Lattice(dimension=dimension, scramble=True, replications=0, seed=7, backend='GAIL')
    measure = Gaussian(distribution, variance=1/2)
    integrand = Keister(measure)
    stopper = CubLattice_g(distribution,abs_tol=abs_tol)
    solution,data = integrate(stopper,integrand,measure,distribution)
    print('%s%s'%(data,bar))


if __name__ == "__main__":
    keister()
