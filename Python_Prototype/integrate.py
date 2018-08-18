from time import time

def integrate(funObj, distribObj, stopCritObj):
    """  Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
     funObj = an object from class fun
     distribObj = an object from class discrete_distribution
     stopcritObj = an object from class stopping_criterion
    """
    # Initialize the accumData object and other crucial objects
    [stopCritObj, dataObj, distribObj] = stopCritObj.stopYet([], funObj, distribObj)
    while  not dataObj.stage == 'done': # the dataObj.stage property tells us where we are in the process
        dataObj = dataObj.updateData(distribObj, funObj);  # compute additional data
        [stopCritObj, dataObj] = stopCritObj.stopYet(stopCritObj, dataObj, funObj)  # update the status of the computation
    solution = dataObj.solution  # assign outputs
    dataObj.timeUsed = time() - dataObj.timeStart
    return solution, dataObj
