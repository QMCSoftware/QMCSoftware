"""
Keister Function Example
    Run Example: python workouts/wo_keister.py
    Save Output: python workouts/wo_keister.py  > outputs/examples/ie_KeisterFun.txt
"""

from qmcpy import integrate
from qmcpy.discrete_distribution import IIDStdUniform, IIDStdGaussian, Lattice, Sobol
from qmcpy.integrand import Keister
from qmcpy.stopping_criterion import CLT, CLTRep, MeanMC_g
from qmcpy.true_measure import Gaussian


def test_distributions_keister(dim, abs_tol):
    """
    Estimate a Keister integral with different discrete sampling distributions.
    """

    # IID Standard Uniform ~ CLT
    integrand = Keister()
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # IID Standard Gaussian ~ CLT
    integrand = Keister()
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()
    
    # IID Standard Uniform ~ MeanMC_g
    integrand = Keister()
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # IID Standard Gaussian ~ MeanMC_g
    integrand = Keister()
    discrete_distrib = IIDStdGaussian(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = MeanMC_g(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # Lattice ~ CLTRep
    integrand = Keister()
    discrete_distrib = Lattice(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()

    # Sobol ~ CLTRelp
    integrand = Keister()
    discrete_distrib = Sobol(rng_seed=7)
    true_measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLTRep(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    data.summarize()


if __name__ == "__main__":
    test_distributions_keister(dim=3, abs_tol=.01)
