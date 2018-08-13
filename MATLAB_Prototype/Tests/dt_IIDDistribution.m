%% dt_IIDDistribution
% doctest for IIDDistribution
%
%  Examples
%
%  Example 1:
%
% >> iid = IIDDistribution
% 
%   iid =***
%     
% 
%          distribData: []
%                state: []
%             nStreams: 1
%               domain: [2x2 double]
%           domainType: 'box'
%            dimension: 2
%     trueDistribution: 'uniform'
%
%  
% >> iid2 = iid.initStreams(1)
% 
%   iid2 =*** 
% 
%   IIDDistribution with properties:
% 
%          distribData: [1x1 struct]
%                state: []
%             nStreams: 1
%               domain: [2x2 double]
%           domainType: 'box'
%            dimension: 2
%     trueDistribution: 'uniform'
%
%
% >> iid2.distribData.stream{1,1}
% 
%     mrg32k3a random stream
%                  Seed: 0
%       NormalTransform: Ziggurat
%
%   
% >> rng(100);
% >> [x, w, a] = iid2.genDistrib(1, 2, 3, [1 2])
% 
%     x =
% 
%         0.7270    0.9387
%         0.4522    0.2360
% 
%     w =
% 
%          1
% 
%     a =
% 
%         0.3333
%


