"""
Keister example
python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
"""

from qmcpy import *
from copy import deepcopy

bar = '\n'+'~'*100+'\n'

def keister(dimension=3, abs_tol=.5):
    print(bar)

    # CubMCCLT
    discrete_distrib = IIDStdUniform(dimension, seed=7)
    integrand = Keister(discrete_distrib)
    stopping_criterion = CubMCCLT(integrand,abs_tol=abs_tol)
    solution,data = stopping_criterion.integrate()
    print('%s%s'%(data,bar))

    # CubQMCCLT
    discrete_distrib = Lattice(dimension, randomize=True, seed=7)
    integrand = Keister(discrete_distrib)
    solution,data = CubQMCCLT(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubMCG
    discrete_distrib = IIDStdUniform(dimension, seed=7)
    integrand = Keister(discrete_distrib)
    solution,data = CubMCG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCLatticeG
    discrete_distrib = Lattice(dimension=dimension, randomize=True, seed=7)
    integrand = Keister(discrete_distrib)
    solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubQMCSobolG
    discrete_distrib = Sobol(dimension=dimension, randomize=True, seed=7)
    integrand = Keister(discrete_distrib)
    solution,data = CubQMCSobolG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubBayesLatticeG
    discrete_distrib = Lattice(dimension=dimension, order='linear', randomize=True)
    integrand = Keister(discrete_distrib)
    solution,data = CubBayesLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('%s%s'%(data,bar))

    # CubBayesNetG
    discrete_distrib = Sobol(dimension=dimension, randomize='LMS', graycode=False)
    integrand = Keister(discrete_distrib)
    solution, data = CubBayesNetG(integrand, abs_tol=abs_tol).integrate()
    print('%s%s' % (data, bar))

if __name__ == "__main__":
    keister(abs_tol=.005)
