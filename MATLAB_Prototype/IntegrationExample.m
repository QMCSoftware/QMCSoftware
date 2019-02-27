%% Integrating a function using our community QMC framework
% An example with Keister's function integrated with respect to the uniform
% distribution over the unit cube
dim = 3; %dimension for the Keister example
measureObj = IIDZMeanGaussian(measure, 'dimension', {dim}, 'variance', {1/2});
distribObj = IIDDistribution('trueD',stdGaussian(measure,'dimension', {dim})); %IID sampling
stopObj = CLTStopping; %stopping criterion for IID sampling using the Central Limit Theorem
[sol, out] = integrate(KeisterFun, measureObj, distribObj, stopObj)
stopObj.absTol = 1e-3; %decrease tolerance
[sol, out] = integrate(KeisterFun, measureObj, distribObj, stopObj)
stopObj.absTol = 0; %impossible tolerance
stopObj.nMax = 1e6; %calculation limited by sample budget
[sol, out] = integrate(KeisterFun, measureObj, distribObj, stopObj)

%A multilevel example of Asian option pricing
stopObj.absTol = 0.01; %increase tolerance
stopObj.nMax = 1e8; %pushing the sample budget back up
measureObj = BrownianMotion(measure,'timeVector', {1/4:1/4:1});
OptionObj = AsianCallFun(measureObj); %4 time steps
distribObj = IIDDistribution('trueD',stdGaussian(measure,'dimension', {4})); %IID sampling
[sol, out] = integrate(OptionObj, measureObj, distribObj, stopObj)

measureObj = BrownianMotion(measure,'timeVector', {1/64:1/64:1});
OptionObj = AsianCallFun(measureObj); %64 time steps
distribObj = IIDDistribution('trueD',stdGaussian(measure,'dimension', {64})); %IID sampling
[sol, out] = integrate(OptionObj, measureObj, distribObj, stopObj)

measureObj = BrownianMotion(measure,'timeVector', {1/4:1/4:1, 1/16:1/16:1, 1/64:1/64:1});
OptionObj = AsianCallFun(measureObj); %multi-level
distribObj = IIDDistribution('trueD',stdGaussian(measure,'dimension', {4, 16, 64})); %IID sampling
[sol, out] = integrate(OptionObj, measureObj, distribObj, stopObj)

