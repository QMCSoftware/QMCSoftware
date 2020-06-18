"""
Keister example
python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
"""

from qmcpy import *
from copy import deepcopy

bar = '\n'+'~'*100+'\n'

def keister(dimension=3, abs_tol=.1):
    print(bar)

    # CubMcClt
    distribution = IIDStdUniform(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    stopping_criterion = CubMcClt(integrand,abs_tol=abs_tol)
    solution,data = stopping_criterion.integrate()
    print('%s%s'%(data,bar))

    # CubQmcClt
    distribution = Lattice(dimension, scramble=True, seed=7, backend='MPS')
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQmcClt(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubMcG
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMcG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQmcLatticeG
    distribution = Lattice(dimension=dimension, scramble=True, seed=7, backend='GAIL')
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQmcLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQmcSobolG
    distribution = Sobol(dimension=dimension, scramble=True, seed=7, backend='QRNG')
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQmcSobolG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

if __name__ == "__main__":
    keister(abs_tol=.005)
