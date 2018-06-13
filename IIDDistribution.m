classdef IIDDistribution < discreteDistribution
%§\mcommentfont Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$ where the $\vx_i$ are IID Uniform on $[0,1]^d$§
properties
   distribData %stream data
   state = [] %not used
   nStreams = 1
end
methods   
   function obj = initStreams(obj,nStreams)
      obj.nStreams = nStreams;
      obj.distribData.stream = RandStream.create('mrg32k3a','NumStreams',nStreams,'CellOutput',true);
   end
      
   function [x, w, a] = genDistrib(obj, nStart, nEnd, n, coordIndex, streamIndex)
      if nargin < 6
         streamIndex = 1;
      end
      nPts = nEnd - nStart + 1; %how many points to be generated
      if strcmp(obj.trueDistribution, 'uniform')
         x = rand(obj.distribData.stream{streamIndex},nPts,numel(coordIndex)); %nodes
      else
         x = randn(obj.distribData.stream{streamIndex},nPts,numel(coordIndex)); %nodes
      end
      w = 1;
      a = 1/n;
   end
end
end
