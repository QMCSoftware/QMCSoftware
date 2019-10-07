"""
Main driver function for Qmcpy.
"""
import copy
from time import time

from .distribution.iid_distribution import IIDDistribution
from .integrand.linear import Linear
from .measures.measures import StdUniform
from .stop.clt import CLT


def integrate(integrand=None, measure=None, distribution=None, \
              stopping_criterion=None):
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
    if integrand is None:
        integrand = Linear()
    if measure is None:
        measure = StdUniform(dimension=[3])
    if distribution is None:
        distribution = IIDDistribution(
            true_distribution=StdUniform(dimension=[3]), seed_rng=7
        )
    if stopping_criterion is None:
        stopping_criterion = CLT(distribution)
    t_start = time()
    # Transform integrands to accept distribution values which can generate
    integrand.transform_variable(measure, distribution)
    while stopping_criterion.stage != "done":
        # the data.stage property tells us where we are in the process
        stopping_criterion.data.update_data(
            distribution, integrand
        )  # compute more data
        stopping_criterion.stop_yet()  # update the status of the computation
    t_total = time() - t_start
    solution = stopping_criterion.data.solution # assign outputs
    data = copy.deepcopy(stopping_criterion.data)
    data.t_total = t_total
    data.integrand = integrand
    data.distribution = distribution
    data.measure = measure
    del stopping_criterion.data
    data.stopping_criterion = stopping_criterion
    return solution, data
