classdef (Abstract) discreteDistribution
%Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$§
properties (Abstract)	
   distribData %information required to generate the distribution
   state %state of the generator
end
properties	
   trueD = stdUniform(measure) %the distribution that the discrete distribution attempts to emulate
end
methods
   function obj = discreteDistribution(varargin) %construct the discrete distribution object
      p = inputParser;
      addParameter(p,'trueD',stdUniform(measure));
      parse(p,varargin{:})
      nObj = numel(p.Results.trueD);
      if nObj == 1
         obj.trueD = p.Results.trueD;
      else
         obj(nObj).trueD = p.Results.trueD(nObj);
         for ii = 1:nObj
            obj(ii).trueD = p.Results.trueD(ii);
         end
      end 
   end
end
methods (Abstract)
   genDistrib(obj, nStart, nEnd, n, coordIndex)
   % nStart = starting value of §$\mcommentfont i$§
   % nEnd = ending value of §$\mcommentfont i$§
   % n = value of §$\mcommentfont n$§ used to determine §$\mcommentfont a_n$§
   % coordIndex = which coordinates in sequence are needed
end
end
