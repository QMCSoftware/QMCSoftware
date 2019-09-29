''' Originally developed in MATLAB by Fred Hickernell. Translated to python by
Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time

from algorithms.distribution import DiscreteDistribution, Measure
from algorithms.function import Fun
from algorithms.stop import StoppingCriterion


def integrate(fun_obj: Fun, measure_obj: Measure,
              distrib_obj: DiscreteDistribution,
              stop_obj: StoppingCriterion) -> tuple:
    """
    Specify and generate values $f(x)$ for $x \in \mathcal{X}$

    Args:
        fun_obj: an object from class Fun
        measure_obj: an object from class maeasure
        distrib_obj: an object from class discrete_distribution
        stop_obj: an object from class stopping_criterion

    Returns:
        (tuple): tuple containing:

            solution (float): estimated value of the integral
            data_obj (AccumData): other information such as number of
            sampling points used to obtain the estimate

    """
    # Initialize the AccumData object and other crucial objects
    fun_obj = fun_obj.transformVariable(measure_obj, distrib_obj)
    stop_obj.stopYet(fun_obj)
    while stop_obj.dataObj.stage != 'done':
        # the dataObj.stage property tells us where we are in the process
        stop_obj.dataObj.updateData(distrib_obj, fun_obj)  # compute more data
        stop_obj.stopYet(fun_obj)  # update the status of the computation
    solution = stop_obj.dataObj.solution  # assign outputs
    stop_obj.dataObj.timeUsed = time() - stop_obj.dataObj._timeStart
    return solution, stop_obj.dataObj
