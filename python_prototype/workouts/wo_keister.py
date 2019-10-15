"""
Keister Function Example
    Run Example:    python workouts/wo_keister.py
    Save Output:    python workouts/wo_keister.py  > outputs/ie_KeisterFun.txt
"""

from qmcpy import integrate
from qmcpy.discrete_distribution import *
from qmcpy.integrand import Keister
from qmcpy.stop import CLT, CLTRep
from qmcpy.true_measure import Gaussian

def test_distributions_keister(dim, abs_tol):

    # IID Standard Uniform
    integrand = Keister()
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # IID Standard Gaussian
    integrand = Keister()
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Lattice
    integrand = Keister()
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

    # Sobol
    integrand = Keister()
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1/2)
    stop = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
    sol, data = integrate(integrand, discrete_distrib, true_measure, stop)
    data.summarize()

if __name__ == '__main__':
    test_distributions_keister(dim=3, abs_tol=.01)
