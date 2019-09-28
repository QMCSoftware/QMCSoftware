''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time

def integrate(funObj,measureObj,distribObj,stopObj):
    '''
    Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
    funObj = an object from class fun
    distribObj = an object from class discrete_distribution
    stopObj = an object from class stopping_criterion
    '''
    # Initialize the accumData object and other crucial objects
    funObj = funObj.transformVariable(measureObj, distribObj)
    stopObj.stopYet(funObj)
    while stopObj.dataObj.stage != 'done': # the dataObj.stage property tells us where we are in the process
        stopObj.dataObj.updateData(distribObj,funObj) # compute additional data
        stopObj.stopYet(funObj) # update the status of the computation
    solution = stopObj.dataObj.solution  # assign outputs
    stopObj.dataObj.timeUsed = time() - stopObj.dataObj._timeStart
    return solution, stopObj.dataObj