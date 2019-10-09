from numpy.random import Generator, PCG64
from copy import deepcopy

from qmcpy import integrate
from qmcpy.integrand import Linear, LazyFunction
from qmcpy.measures import CustomIID, StdUniform
from qmcpy.distribution import IIDDistribution
from qmcpy.stop import CLT


def custom_lazy_generator():
    dim = 3
    lambda_ = 10
    # Create an appropriote Poisson generator
    numpy_gen = Generator(PCG64(7))  # seed is 7
    poisson_wrapper = lambda *size: numpy_gen.poisson(lam=lambda_, size=size)
    # Use qmcpy for integration
    integrand = Linear()
    measure = CustomIID(dimension=[dim], generator=poisson_wrapper)
    distribution = IIDDistribution(true_distribution=deepcopy(measure))
    stop = CLT(distribution, abs_tol=0.01)
    # exact_solution = dim*lambda_
    sol, data_obj = integrate(integrand, measure, distribution, stop)
    data_obj.summarize()


def custom_lazy_integrand():
    dim = 5
    integrand = LazyFunction(custom_fun=lambda x, coordIdx: (5 * x).sum(1))
    measure = StdUniform(dimension=[dim])
    distribution = IIDDistribution(
        true_distribution=StdUniform(dimension=[dim]), seed_rng=7)
    stop = CLT(distribution, abs_tol=0.01)
    sol, data = integrate(integrand, measure, distribution, stop)
    # exact_solution = dim*(5/2)
    data.summarize()


if __name__ == "__main__":
    custom_lazy_generator()
    print()
    custom_lazy_integrand()
