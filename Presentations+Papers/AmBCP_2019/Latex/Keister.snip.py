d = 2
stopObj = CLTStopping(nInit=16,
                      absTol=.5)
measureObj = measure().\
    iid_zmean_gaussian(dimension=[d],
                       variance=[.5])
distribObj = IIDDistribution(
    trueD=measure().std_gaussian(
        dimension=[d]))
sol, out = integrate(KeisterFun(),
    measureObj, distribObj, stopObj)
