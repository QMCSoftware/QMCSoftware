"""
Workout for QuickConstruct integrand class.
"""
from qmcpy import integrate
from qmcpy.integrand import QuickConstruct
from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.true_measure import Uniform
from qmcpy.stopping_criterion import CLT


def quick_construct_integrand(abs_tol):
    """
    Demonstrate how to use QuickConstruct to define a QMCPy integrand.

    Args:
        abs_tol (float): absolute tolerance

    Returns:
        None

    """
    dim = 5
    integrand = QuickConstruct(custom_fun=lambda x: (5 * x).sum(1))
    discrete_distrib = IIDStdUniform(rng_seed=7)
    true_measure = Uniform(dim)
    stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=abs_tol)
    _, data = integrate(integrand, true_measure, discrete_distrib, stopping_criterion)
    # exact_solution = dim*(5/2)
    data.summarize()


if __name__ == "__main__":
    quick_construct_integrand(abs_tol=.01)
