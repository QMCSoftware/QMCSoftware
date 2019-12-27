""" Main driver function for QMCPy. """

from time import process_time

from ._util import DimensionError
from .discrete_distribution import IIDStdUniform
from .stopping_criterion import CLT


def integrate(integrand, true_measure, discrete_distrib=None, stopping_criterion=None):
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
    if not discrete_distrib: discrete_distrib = IIDStdUniform(rng_seed=7)
    if not stopping_criterion:
        stopping_criterion = CLT(discrete_distrib, true_measure, abs_tol=0.01)
    # Check TrueMeasure and Integrand matching dimensions
    if not all(integrand.dimension == true_measure.dimension):
        raise DimensionError('The integrand and true measure should have the same dimension(s)')
    t_start = process_time()
    # Transform integrands to accept distribution values which can generate
    true_measure.transform(integrand, discrete_distrib)
    while stopping_criterion.stage != "done":
        # the data.stage property tells us where we are in the process
        stopping_criterion.data.update_data(
            integrand, true_measure)  # compute more data
        stopping_criterion.stop_yet()  # update the status of the computation
    solution = stopping_criterion.data.solution  # assign outputs
    cpu_time = process_time() - t_start
    data_obj_final = stopping_criterion.data.complete(cpu_time, integrand,
                                                      discrete_distrib,
                                                      true_measure,
                                                      stopping_criterion)
    return solution, data_obj_final
