%% Example
stopObj = CLTStopping %stopping criterion
distribObj = IIDDistribution; %IID sampling with uniform distribution
[sol, out] = integrate(KeisterFun, distribObj, stopObj)
stopObj.absTol = 1e-3; %decrease tolerance
[sol, out] = integrate(KeisterFun, distribObj, stopObj)

distribObj.trueDistribution = 'normal'; %Change to normal distribution
stopObj.absTol = 0.01; %increase tolerance
OptionObj = AsianCallFun(4) %4 time steps
[sol, out] = integrate(OptionObj, distribObj, stopObj)
OptionObj = AsianCallFun(64) %single level, 64 time steps
[sol, out] = integrate(OptionObj, distribObj, stopObj)
OptionObj = AsianCallFun([4 4 4]) %multilevel, 64 time steps, faster
[sol, out] = integrate(OptionObj, distribObj, stopObj)