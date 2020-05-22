classdef IIDDistribution < discreteDistribution
%§\mcommentfont Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$§
%§\mcommentfont    where the $\vx_i$ are IID uniform on $[0,1]^d$ or IID standard Gaussian §
properties
   distribData %stream data
   state = [] %not used
end

methods   
   
   function obj = IIDDistribution(varargin)
      obj = obj@discreteDistribution(varargin{:});
      nObj = numel(obj);
      p = inputParser;
      p.KeepUnmatched = true;
      addParameter(p,'distribData',cell(1,nObj));
      parse(p,varargin{:})
      [obj.distribData] = p.Results.distribData{:};
   end
      
   function obj = initStreams(obj)
      nObj = numel(obj);
      temp = RandStream.create('mrg32k3a','NumStreams',nObj,'CellOutput',true);
      for ii = 1:nObj
         obj(ii).distribData.stream = temp{ii};
      end
   end
      
   function [x, w, a] = genDistrib(obj, nStart, nEnd, n, coordIndex)
      if nargin < 5
         coordIndex = 1:obj.trueD.dimension;
      end
      nPts = nEnd - nStart + 1; %how many points to be generated
      if strcmp(obj.trueD.measureName, 'stdUniform') %generate uniform points
         x = rand(obj.distribData.stream,nPts,numel(coordIndex)); %nodes
      elseif strcmp(obj.trueD.measureName, 'stdGaussian') %standard normal points
         x = randn(obj.distribData.stream,nPts,numel(coordIndex)); %nodes
      else
         error('Distribution not recognized')
      end
      w = 1;
      a = 1/n;
   end
end
end
