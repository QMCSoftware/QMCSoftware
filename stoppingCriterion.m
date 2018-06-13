classdef (Abstract) stoppingCriterion
%§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties
   absTol = 1e-2 %absolute tolerance, §$\mcommentfont d$§
   relTol = 0 %absolute tolerance, §$\mcommentfont d$§
end
properties (Abstract)
   discDistAllowed %which discrete distributions are supported
   decompTypeAllowed %which decomposition types are supported
end
methods (Abstract)
	stopYet(obj, distribObj)
   % oldData = data or summary of data computed already
   % newData = new (summary) data 
end
end

