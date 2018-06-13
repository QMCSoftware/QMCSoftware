%% Example
stopObj = CLTStopping
distribObj = IIDDistribution;
[sol, out] = integrate(KeisterFun, distribObj, stopObj)
stopObj.absTol = 1e-3;
[sol, out] = integrate(KeisterFun, distribObj, stopObj)

distribObj.trueDistribution = 'normal';
stopObj.absTol = 0.02;
OptionObj = AsianCallFun(4) %4 time steps
[sol, out] = integrate(OptionObj, distribObj, stopObj)
OptionObj = AsianCallFun(64) %single level, 64 time steps
[sol, out] = integrate(OptionObj, distribObj, stopObj)
OptionObj = AsianCallFun([4 4 4]) %multilevel, 64 time steps
[sol, out] = integrate(OptionObj, distribObj, stopObj)