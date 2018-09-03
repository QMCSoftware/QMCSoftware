%% dt_integrate
% doctest for integrate
%
%  Examples
%
%  Example 1: integrate over given default instances of KeisterFun,
%  IIDDistribution, and CLTStopping.
%
% >> funObj = KeisterFun; distribObj = IIDDistribution; stopObj = CLTStopping; 
% >> [solution, dataObj] = integrate(funObj, distribObj, stopObj)
% 
% solution =
% 
%     0.4310
% 
% dataObj = 
%
%  meanVarData with properties:
% 
%            muhat: 0.4310
%           sighat: 0.2611
%           nSigma: 1024
%              nMu: 6516
%         solution: 0.4310
%            stage: 'done'
%            prevN: 1024
%            nextN: 7540
%         timeUsed: 0.00***
%     nSamplesUsed: 7540
%       errorBound: [0.4210 0.4410]
%            costF: ***e-04
%
%
%  Example 2: integrate over given default instances of KeisterFun and
%  IIDDistribution, but overriding default absTol in the instance of
%  CLTStopping.
%
% >> stopObj.absTol = 1e-3;
% >> [solution, dataObj] = integrate(funObj, distribObj, stopObj)
% 
% solution =
% 
%     0.4253
% 
% dataObj =
%
%   meanVarData with properties:
%
%            muhat: 0.4253
%           sighat: 0.2611
%           nSigma: 1024
%              nMu: 651522
%         solution: 0.4253
%            stage: 'done'
%            prevN: 1024
%            nextN: 652546
%         timeUsed: 0.0***
%     nSamplesUsed: 652546
%       errorBound: [0.4243 0.4263]
%            costF: 0.0***
%
%
%  Example 3: integrate over given default instances of KeisterFun and
%  IIDDistribution, but deactivating stopping condition with absTol and 
%  using nMax instead in the instance of CLTStopping.
%
% >> stopObj.absTol = 0;  stopObj.nMax = 1e6; 
% >> [solution, dataObj] = integrate(funObj, distribObj, stopObj)
% 
% solution =
% 
%     0.4252
% 
% dataObj =*** 
%
%            muhat: 0.4252
%           sighat: 0.2611
%           nSigma: 1024
%              nMu: 998976
%         solution: 0.4252
%            stage: 'done'
%            prevN: 1024
%            nextN: 1000000
%         timeUsed: 0.0***
%     nSamplesUsed: 1000000
%       errorBound: [0.4244 0.4260]
%            costF: 0.0***

