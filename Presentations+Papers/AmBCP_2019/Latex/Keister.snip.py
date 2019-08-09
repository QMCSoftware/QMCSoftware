dim = 2
stopObj = CLTStopping(nInit=16,absTol=.5)
measureObj = measure().IIDZMeanGaussia(
    dimension=[dim],variance=[.5])
distribObj = IIDDistribution(
    trueD=measure().stdGaussian(
        dimension=[dim]))
sol,out = integrate(KeisterFun(),
    measureObj,distribObj,stopObj)