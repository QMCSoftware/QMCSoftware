from time import time

def integrate(funObj, distribObj, stopCritObj, datObj=[]):
    """  Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
     funObj = an object from class fun
     distribObj = an object from class discrete_distribution
     stopcritObj = an object from class stopping_criterion
    """
    # Initialize the accumData object and other crucial objects
    [datObj, distribObj] = stopCritObj.stopYet(datObj, funObj, distribObj)
    while not datObj.stage == 'done': # the datObj.stage property tells us where we are in the process
        datObj.updateData(distribObj, funObj)  # compute additional data
        [datObj, distribObj] = stopCritObj.stopYet(datObj, funObj, distribObj)  # update the status of the computation
    solution = datObj.solution  # assign outputs
    datObj.timeUsed = time() - datObj.timeStart
    return solution, datObj


import addpath
addpath
from Keister import Keister as Keister
from IID import IID as IID
from CLT import CLT as CLT
from meanVar import meanVar as meanVar
funObj = Keister()
distribObj = IID()
stopObj = CLT()
#datObj = meanVar()
[solution, datObj] = integrate(funObj, distribObj, stopObj)
print(solution)
print(datObj.__dict__)


stopObj.absTol = 1e-3
[solution, datObj] = integrate(funObj, distribObj, stopObj)
print(solution)
print(datObj.__dict__)


stopObj.absTol = 0
stopObj.nMax = 1e6
[solution, datObj] = integrate(funObj, distribObj, stopObj)
print(solution)
print(datObj.__dict__)


