%% dt_meanVarData
% doctest for meanVarData
%
%  Examples
%
%  Example 1: Create a meanVarData instance.
%
% >> mvd = meanVarData
% 
%   mvd =***
%     
%            muhat: []
%           sighat: []
%           nSigma: []
%              nMu: []
%         solution: NaN
%            stage: 'begin'
%            prevN: []
%            nextN: []
%         timeUsed: []
%     nSamplesUsed: []
%       errorBound: [-Inf Inf]
%            costF: []
%
%
%  Example 2: Constructor
%
% >> iid = IIDDistribution; iid = iid.initStreams(1); 
% >> rng(100); [x, w, a] = iid.genDistrib(1, 2, 3, [1 2]);
% >> kf = KeisterFun; y = kf.f([1,2; 3 4], [1 2]);
% >> mvd.updateData(iid, kf)

