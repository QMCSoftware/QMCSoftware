"""
Keister example
python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
"""

from qmcpy import *
from copy import deepcopy

bar = '\n'+'~'*100+'\n'

def keister(dimension=3, abs_tol=.1):
    print(bar)

    # CubMCCLT
    distribution = IIDStdUniform(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    stopping_criterion = CubMCCLT(integrand,abs_tol=abs_tol)
    solution,data = stopping_criterion.integrate()
    print('%s%s'%(data,bar))

    # CubQMCCLT
    distribution = Lattice(dimension, randomize=True, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubMCG
    distribution = IIDStdGaussian(dimension, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubMCG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCLatticeG
    distribution = Lattice(dimension=dimension, randomize=True, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCSobolG
    distribution = Sobol(dimension=dimension, randomize=True, seed=7)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubQMCSobolG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubBayesLatticeG
    distribution = Lattice(dimension=dimension, order='linear', randomize=False)
    measure = Gaussian(distribution, covariance=1./2)
    integrand = Keister(measure)
    solution,data = CubBayesLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

if __name__ == "__main__":
    keister(abs_tol=.005)
