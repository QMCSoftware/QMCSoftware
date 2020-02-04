""" Main integration method acting on QMCPy objects """

from .distribution._distribution import Distribution
from .measure._measure import Measure
from .integrand._integrand import Integrand
from .util import DimensionError
from time import process_time


def integrate(integrand, measure, distribution, stopping_criterion):
    """
    Specify and compute integral of :math:`f(\\boldsymbol{x})` for \
    :math:`\\boldsymbol{x} \\in \\mathcal{X}`.

    Args:
        integrand (Integrand): an Integrand object
        measure (Measure): a Measure object
        distribution (Distribution): a Distribution object
        stopping_criterion (StoppingCriterion): a StoppingCrition object

    Returns:
        solution (float): estimated value of the integral
        data (AccumData): houses input and integration process data
            Includes:
                self.integrand (origianl Integrand): origianl integrand
                self.measure (original Measure): origianl measure
                self.stopping_criterion (origianl StoppingCriterion)
            Note: Calling print(data) outputs a string of all parameters
                        
    """
    # Check matching dimensions for integrand, measure, and distribution
    flag = 0
    try:
        if isinstance(distribution,Distribution):
            distrib_dims = distribution.dimension
            measure_dims = measure.dimension
            integrand_dims = integrand.dimension
        else: # multi-level
            distrib_dims = distribution.dimensions
            measure_dims = measure.dimensions
            integrand_dims = integrand.dimensions
        if distrib_dims != measure_dims or distrib_dims != integrand_dims:
            flag = 1
    except:
        flag = 1
    if flag == 1: # mismatching dimensions or incorrectlly constructed objects
        raise DimensionError('''
                distribution, measure, and integrand dimensions do not match. 
                For multi-level problems ensure distribution, measure, and integrand
                are MultiLevelConstructor instances. ''') 
    # Start Integration
    t_start = process_time()
    while stopping_criterion.stage != "done":
        # the data.stage property tells us where we are in the process
        stopping_criterion.data.update_data(integrand, measure)  # compute more data
        stopping_criterion.stop_yet()  # update the status of the computation
    solution = stopping_criterion.data.solution  # assign outputs
    cpu_time = process_time() - t_start
    stopping_criterion.data.complete(\
        cpu_time, integrand, distribution, measure, stopping_criterion)
    return solution, stopping_criterion.data
