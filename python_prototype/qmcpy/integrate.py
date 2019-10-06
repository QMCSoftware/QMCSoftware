"""
Main driver function for Qmcpy.
"""
from time import time

from .integrand.keister import Keister
from .measures.measures import IIDZeroMeanGaussian, Sobol
from .distribution.quasi_random import QuasiRandom
from .stop.clt_rep import CLTRep

def integrate(integrand=Keister(), \
              measure=IIDZeroMeanGaussian(dimension=[3], variance=[1 / 2]), \
              distribution=QuasiRandom(true_distribution=Sobol(dimension=[3])), \
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
    
    if stopping_criterion is None:
        stopping_criterion = CLTRep(distribution)
    t_start = time()
    # Transform integrands to accept distribution values which can generate
    integrand.transform_variable(measure, distribution)
    while stopping_criterion.stage != 'done':
        # the data_obj.stage property tells us where we are in the process
        stopping_criterion.data_obj.update_data(distribution, integrand)  # compute more data
        stopping_criterion.stop_yet()  # update the status of the computation
    solution = stopping_criterion.data_obj.solution  # assign outputs
    stopping_criterion.data_obj.t_total = time() - t_start
    return solution, stopping_criterion.data_obj
