d = 2
stopObj = CLTStopping(nInit=16,
                      absTol=.5)
measureObj = measure().\
    IIDZMeanGaussian(dimension=[d],
                     variance=[.5])
distribObj = IIDDistribution(
    trueD=measure().stdGaussian(
        dimension=[d]))
sol, out = integrate(KeisterFun(),
    measureObj, distribObj, stopObj)
