from time import time

def integrate(funObj, distribObj, stopCritObj, datObj):
    """  Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
     funObj = an object from class fun
     distribObj = an object from class discrete_distribution
     stopcritObj = an object from class stopping_criterion
    """
    # Initialize the accumData object and other crucial objects
    [stopCritObj, dataObj, distribObj] = stopCritObj.stopYet(datObj, funObj, distribObj)
    while  not dataObj.stage == 'done': # the dataObj.stage property tells us where we are in the process
        dataObj = dataObj.updateData(distribObj, funObj);  # compute additional data
        [stopCritObj, dataObj, _] = stopCritObj.stopYet(datObj, funObj)  # update the status of the computation
    solution = dataObj.solution  # assign outputs
    dataObj.timeUsed = time() - dataObj.timeStart
    return solution, dataObj


import addpath
addpath
from Keister import Keister as Keister
from IID import IID as IID
from CLT import CLT as CLT
from meanVar import meanVar as meanVar
funObj = Keister()
distribObj = IID()
stopObj = CLT()
datObj = meanVar()
[solution, dataObj] = integrate(funObj, distribObj, stopObj, datObj)
print(solution)
print(dataObj.__dict__)
