classdef (Abstract) stoppingCriterion
% Decide when to stop a
properties
   absTol = 1e-2 %absolute tolerance, §$\mcommentfont d$§
   relTol = 0 %absolute tolerance, §$\mcommentfont d$§
   nInit = 1024 %initial sample size
   nMax = 1e8 %maximum number of samples allowed
end
properties (Abstract)
   discDistAllowed %which discrete distributions are supported
   decompTypeAllowed %which decomposition types are supported
end
methods (Abstract)
	stopYet(obj, distribObj)
   % distribObj = data or summary of data computed already
end
end

