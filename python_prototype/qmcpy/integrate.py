""" Main driver function for QMCPy. """

import copy
from time import time

from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.stopping_criterion import CLT


def integrate(integrand, true_measure, discrete_distrib=None, stopping_criterion=None):
    """Specify and compute integral of :math:`f(\\boldsymbol{x})` for \
    :math:`\\boldsymbol{x} \\in \\mathcal{X}`.

    Args:
        integrand (Integrand): an object from class Integrand
        true_measure (TrueMeasure): an object from class TrueMeasure
        discrete_distrib (DiscreteDistribution): an object from class \
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
    if not stopping_criterion:
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=.01)

    t_start = time()
    # Transform integrands to accept distribution values which can generate
    true_measure.transform(integrand, discrete_distrib)
    while stopping_criterion.stage != "done":
        # the data.stage property tells us where we are in the process
        stopping_criterion.data.update_data(
            integrand, true_measure)  # compute more data
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
