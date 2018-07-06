classdef (Abstract) discreteDistribution
%Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$§
properties (Abstract)	
   distribData %information required to generate the distribution
   state %state of the generator
   nStreams
end
properties	
   domain = [0 0; 1 1]; %domain of the discrete distribution, §$\mcommentfont \cx$§
   domainType = 'box' %domain of the discrete distribution, §$\mcommentfont \cx$§
   dimension = 2 %dimension of the domain, §$\mcommentfont d$§
   trueDistribution = 'uniform' %name of the distribution that the discrete distribution attempts to emulate
end
methods (Abstract)
   genDistrib(obj, nStart, nEnd, n, coordIndex)
   % nStart = starting value of §$\mcommentfont i$§
   % nEnd = ending value of §$\mcommentfont i$§
   % n = value of §$\mcommentfont n$§ used to determine §$\mcommentfont a_n$§
   % coordIndex = which coordinates in sequence are needed
end
end
