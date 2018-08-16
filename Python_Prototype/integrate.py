import time

def integrate(funObj, distribObj, stopCritObj):
    #Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
    # funObj = an object from class fun
    # distribObj = an object from class discrete_distribution
    # stopcritObj = an object from class stopping_criterion

    # Initialize the accumData object and other crucial objects
    dataObj, distribObj = stopCritObj.stopYet(funObj, distribObj)
    while dataObj.stage != 'done': #the dataObj.stage property tells us where we are in the process
        dataObj = dataObj.updateData(distribObj, funObj) # compute additional data
        stopCritObj, dataObj = stopCritObj.stopYet(funObj, distribObj, dataObj) # update the status of the computation
    solution = dataObj.solution # assign outputs
    dataObj.timeUsed = time.time() - dataObj.timeStart
    return solution, dataObj

if __name__ == "__main__":
    from CLTStopping import CLTStopping
    from IIDDistribution import IIDDistribution
    from KeisterFun import KeisterFun
    stopObj = CLTStopping()  # stopping criterion for IID sampling using the Central Limit Theorem
    distribObj = IIDDistribution()
    k = KeisterFun()  # IID sampling with uniform distribution
    sol, out = integrate(k, distribObj, stopObj)
    print(sol,"\n\n",out)
