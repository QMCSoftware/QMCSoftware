classdef (Abstract) accumData
% Accumulated data required in the computation of the integral
properties
   solution = NaN %solution
   stage = 'begin' %stage of the computation, becomes 'done' when finished
   prevN %new data will be based on (quasi-)random vectors indexed by
   nextN %prevN + 1 to nextN
   timeStart %starting time
   timeUsed %time used so far
   nSamplesUsed %number of samples used so far
   errorBound = [-Inf Inf] %error bound on the solution
   costF %time required to compute function values
end
methods (Abstract)
	updateData(obj, distribObj, fun_obj, decompType)
end
end
