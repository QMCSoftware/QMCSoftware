"""
Main driver function for Qmcpy.
"""
import copy
from time import time

from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.stop import CLT


def integrate(integrand, true_measure, discrete_distrib=None, stopping_criterion=None):
    """Specify and compute integral of :math:`f(\\mathbf{x})` for \
    :math:`\\mathbf{x} \\in \\mathcal{X}`.

    Args:
        integrand (Integrand): an object from class Integrand
        measure (Measure): an object from class Measure
        distribution (DiscreteDistribution): an object from class \
            DiscreteDistribution
        stopping_criterion (StoppingCriterion): an object from class \
            StoppingCriterion

    Returns:
        tuple: tuple containing:

            **solution** (:obj:`float`): estimated value of the integral

            **data** (:obj:`AccumData`): other information such as number of \
                sampling points used to obtain the estimate

    """

    # Default some arguments
    if not discrete_distrib: discrete_distrib = IIDStdUniform()
    if not stopping_criterion: stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=.01)

    t_start = time()
    # Transform integrands to accept distribution values which can generate
    true_measure.transform_generator(discrete_distrib)
    while stopping_criterion.stage != "done":
        # the data.stage property tells us where we are in the process
        stopping_criterion.data.update_data(
            true_measure, integrand)  # compute more data
        stopping_criterion.stop_yet()  # update the status of the computation
    time_total = time() - t_start
    solution = stopping_criterion.data.solution  # assign outputs
    data = copy.deepcopy(stopping_criterion.data)
    data.time_total = time_total
    data.integrand = integrand
    data.discrete_distrib = discrete_distrib
    data.true_measure = true_measure
    del stopping_criterion.data
    data.stopping_criterion = stopping_criterion
    return solution, data
