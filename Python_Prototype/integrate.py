from time import time

def integrate(funObj, distribObj, stopCritObj, datObj=[]):
    """  Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
     funObj = an object from class fun
     distribObj = an object from class discrete_distribution
     stopcritObj = an object from class stopping_criterion
    """
    # Initialize the accumData object and other crucial objects
    funObj.transformVariable(distribObj)
    [datObj, distribObj] = stopCritObj.stopYet(datObj, funObj, distribObj)
    while not datObj.stage == 'done': # the datObj.stage property tells us where we are in the process
        datObj.updateData(distribObj, funObj)  # compute additional data
        [datObj, distribObj] = stopCritObj.stopYet(datObj, funObj, distribObj)  # update the status of the computation
    solution = datObj.solution  # assign outputs
    datObj.timeUsed = time() - datObj.timeStart
    return solution, datObj

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("Tests/dt_integrate.py")
    print("\n"+str(x))



