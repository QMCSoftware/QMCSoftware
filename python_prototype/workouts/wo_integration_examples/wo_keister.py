"""
Keister Function Example
    Save Output:
        python workouts/wo_integration_examples/wo_keister.py  > outputs/wo_integration_examples/ie_KeisterFun.txt
"""

from qmcpy import *


def test_distributions_keister(dim, abs_tol):
    """
    Estimate a Keister integral with different discrete sampling distributions.
    """

    # IID Standard Uniform ~ CLT
    integrand = Keister(dim)
    distrib = IIDStdUniform(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLT(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # IID Standard Gaussian ~ CLT
    integrand = Keister(dim)
    distrib = IIDStdGaussian(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLT(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # IID Standard Uniform ~ MeanMC_g
    integrand = Keister(dim)
    distrib = IIDStdUniform(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = MeanMC_g(distrib, measure,
                                  abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # IID Standard Gaussian ~ MeanMC_g
    integrand = Keister(dim)
    distrib = IIDStdGaussian(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = MeanMC_g(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # Lattice ~ CLTRep
    integrand = Keister(dim)
    distrib = Lattice(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLTRep(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)

    # Sobol ~ CLTRep
    integrand = Keister(dim)
    distrib = Sobol(rng_seed=7)
    measure = Gaussian(dimension=dim, variance=1 / 2)
    stopping_criterion = CLTRep(distrib, measure, abs_tol=abs_tol)
    _, data = integrate(integrand, measure,
                        distrib, stopping_criterion)
    print(data)


if __name__ == "__main__":
    test_distributions_keister(dim=3, abs_tol=.01)
