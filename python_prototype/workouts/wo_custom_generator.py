from numpy.random import Generator,PCG64
from copy import deepcopy

from qmcpy import integrate
from qmcpy.integrand import Linear
from qmcpy.measures import CustomIID
from qmcpy.distribution import IIDDistribution
from qmcpy.stop import CLT

def custom_poisson_generator_example(dim,lambda_):
    # Create an appropriote Poisson generator
    numpy_gen = Generator(PCG64(7)) # seed is 7
    poisson_wrapper = lambda *size: numpy_gen.poisson(lam=lambda_,size=size)
    # Use qmcpy for integration
    integrand = Linear()
    measure = CustomIID(dimension=[dim], generator=poisson_wrapper)
    distribution = IIDDistribution(true_distribution=deepcopy(measure))
    stop = CLT(distribution, abs_tol=.1)
    # exact_solution = dim*lambda_
    return integrate(integrand, measure, distribution, stop)

if __name__ == '__main__':
    sol,data_obj = custom_poisson_generator_example(3,5)
    data_obj.summarize

    from qmcpy.measures import StdGaussian
    numpy_gen = Generator(PCG64(7)) # seed is 7
    poisson_wrapper = lambda *size: numpy_gen.poisson(lam=4,size=size)
    # Use qmcpy for integration
    integrand = Linear()
    measure = StdGaussian(dimension=[3])
    distribution = IIDDistribution(true_distribution=CustomIID(dimension=[3], generator=poisson_wrapper))
    stop = CLT(distribution, abs_tol=.1)
    # exact_solution = dim*lambda_
    sol,dataObj = integrate(integrand, measure, distribution, stop)
    print(sol)

