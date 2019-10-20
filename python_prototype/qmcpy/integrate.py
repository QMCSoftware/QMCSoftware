""" Main driver function for QMCPy. """

import copy
from time import time

from qmcpy.discrete_distribution import IIDStdUniform
from qmcpy.integrand.linear import Linear
from qmcpy.stopping_criterion import CLT
from qmcpy.true_measure import Uniform


def integrate(integrand=None, true_measure=None, discrete_distrib=None, stopping_criterion=None):
    """Specify and compute integral of :math:`f(\\boldsymbol{x})` for \
    :math:`\\boldsymbol{x} \\in \\mathcal{X}`.

    Args:
        integrand (Integrand): an object from class Integrand. If None (default),
            sum of two variables defined on unit square is used.
        true_measure (TrueMeasure): an object from class TrueMeasure. If None \
        (default), standard uniform distribution is used.
        discrete_distrib (DiscreteDistribution): an object from class \
            DiscreteDistribution. If None (default), IID standard uniform \
            distribution is used.
        stopping_criterion (StoppingCriterion): an object from class \
            StoppingCriterion.  If None (default), criterion based on central \
            limit theorem with absolute tolerance equal to 0.01 is used.

    Returns:
        tuple: tuple containing:

            **solution** (:obj:`float`): estimated value of the integral

            **data** (:obj:`AccumData`): input data and information such as \
                number of sampling points and run time used to obtain solution

    """

    # Default some arguments
    if not integrand: integrand = Linear()
    if not true_measure: true_measure = Uniform(2)
    if not discrete_distrib: discrete_distrib = IIDStdUniform()
    if not stopping_criterion:
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=0.01)

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
