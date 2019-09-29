''' Originally developed in MATLAB by Fred Hickernell. Translated to python by
Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time


def integrate(fun_obj, measure_obj, distrib_obj, stop_obj):
    '''
    Specify and generate values $f(x)$ for $x \in \mathcal{X}$

    :param fun_obj: an object from class fun
    :param measure_obj: an object from class maeasure
    :param distrib_obj: an object from class discrete_distribution
    :param stop_obj:  an object from class stopping_criterion
    :return: None
    '''
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
