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
% dataObj =*** 
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