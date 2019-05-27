from time import time

def integrate(funObj,measureObj,distribObj,stopCritObj):
    '''
    Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
    funObj = an object from class fun
    distribObj = an object from class discrete_distribution
    stopcritObj = an object from class stopping_criterion
    '''
    # Initialize the accumData object and other crucial objects
    funObj = funObj.transformVariable(measureObj, distribObj)
    dataObj,distribObj = stopCritObj.stopYet(funObj=funObj,distribObj=distribObj)
    while dataObj.stage != 'done': # the dataObj.stage property tells us where we are in the process
        dataObj = dataObj.updateData(dataObj,distribObj,funObj) # compute additional data
        dataObj,distribObj = stopCritObj.stopYet(dataObj,funObj) # update the status of the computation
    solution = dataObj.solution  # assign outputs
    datObj.timeUsed = time() - dataObj.timeStart
    return solution, dataObj

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("Tests/dt_integrate.py")
    print("\n"+str(x))