from qmcpy import integrate
from qmcpy.integrand import QuickConstruct
from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.true_measure import Uniform
from qmcpy.stop import CLT

def quick_construct_integrand(abs_tol):
    dim = 5
    integrand = QuickConstruct(custom_fun=lambda x: (5 * x).sum(1))
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Uniform(dim)
    stop = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    sol, data = integrate(integrand, true_measure, discrete_distrib, stop)
    # exact_solution = dim*(5/2)
    data.summarize()

if __name__ == "__main__":
    quick_construct_integrand(abs_tol=.01)
