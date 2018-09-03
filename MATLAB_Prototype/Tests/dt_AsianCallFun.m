%% dt_AsianCallFun
% doctest for AsianCallFun
%
%  Examples
%
%  Example 1: Create a AsianCallFun instance.
%
% >> acf = AsianCallFun
%
%     acf =***
% 
%           volatility: 0.5000
%                   S0: 30
%                    K: 25
%                    T: 1
%                    A: []
%                 tVec: []
%               dimFac: 1
%               domain: [2x2 double]
%           domainType: 'box'
%            dimension: 2
%          distribType: 'uniform'
%         nominalValue: 0
%
%
%  Example 2: Invoke method f on the AsianCallFun instance.
%
%  >> acf = AsianCallFun(4)
% 
%     acf =***
% 
%           volatility: 0.5000
%                   S0: 30
%                    K: 25
%                    T: 1
%                    A: [4x4 double]
%                 tVec: [0.2500 0.5000 0.7500 1]
%               dimFac: 0
%               domain: [2x2 double]
%           domainType: 'box'
%            dimension: 4
%          distribType: 'uniform'
%         nominalValue: 0
%
%
%  >> y = acf.f([-0.1894  0.2209  1.6796  0.1918;  -1.4426  0.7414  -0.6234 0.7707], [1 2 3 4])
% 
%   y =
%
%         4.3603
%         2.0002