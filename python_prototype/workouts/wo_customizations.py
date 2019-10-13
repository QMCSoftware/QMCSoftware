from numpy.random import Generator, PCG64
from copy import deepcopy

from qmcpy import integrate
from qmcpy.integrand import Linear, QuickConstruct
from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.true_distribution import Uniform
from qmcpy.stop import CLT

def quick_construct_integrand():
    dim = 5
    integrand = QuickConstruct(custom_fun=lambda x: (5 * x).sum(1))
    discrete_distrib = IIDStdUniform()
    true_distrib = Uniform(dim)
    stop = CLT(discrete_distrib, true_distrib , abs_tol=0.01)
    sol, data = integrate(integrand, discrete_distrib, true_distrib, stop)
    # exact_solution = dim*(5/2)
    data.summarize()


if __name__ == "__main__":
    quick_construct_integrand()

