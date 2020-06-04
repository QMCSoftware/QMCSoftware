%% dt_KeisterFun
% doctest for KeisterFun
%
%  Examples
%
%  Example 1: Create a KisterFun instance.
%
% >> kf = KeisterFun
%
%   kf = 
%
%    KeisterFun with properties:
%     
%     domain: [2x2 double]
%     domainType: 'box'
%     dimension: 2
%     distribType: 'uniform'
%     nominalValue: 0    
%
%
%  Example 2: Invoke method f on the KisterFun instance.
%
%  >> y = kf.f([1,2; 3 4], [1 2])
% 
%   y =
%
%    -0.0042
%     0.0000
%
%
