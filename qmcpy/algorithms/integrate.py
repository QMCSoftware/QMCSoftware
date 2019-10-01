from time import time

from algorithms.distribution import DiscreteDistribution, Measure
from algorithms.integrand import Integrand
from algorithms.stop import StoppingCriterion


def integrate(fun_obj: Integrand, measure_obj: Measure,
              distrib_obj: DiscreteDistribution,
              stop_obj: StoppingCriterion) -> tuple:
    """
    Specify and generate values :math:`f(\mathbf{x})` for :math:`\mathbf{x} \in \mathcal{X}`

    Args:
        fun_obj: an object from class Integrand
        measure_obj: an object from class Measure
        distrib_obj: an object from class DiscreteDistribution
        stop_obj: an object from class StoppingCriterion

    Returns:
        (tuple): tuple containing:

            solution (float): estimated value of the integral

            data_obj (AccumData): other information such as number of
            sampling points used to obtain the estimate

    """
    t_start = time()
    # Transform integrands to accept distribution values which can generate
    fun_obj = fun_obj.transform_variable(measure_obj, distrib_obj)
    while stop_obj.stage != 'done':
        # the data_obj.stage property tells us where we are in the process
        stop_obj.data_obj.update_data(distrib_obj, fun_obj)  # compute more data
        stop_obj.stopYet(fun_obj)  # update the status of the computation
    solution = stop_obj.data_obj.solution  # assign outputs
    stop_obj.data_obj.t_total = time() - t_start
    return solution, stop_obj.data_obj
