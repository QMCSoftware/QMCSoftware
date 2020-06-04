classdef (Abstract) accumData
% Accumulated data required in the computation of the integral
properties
   solution = NaN %solution
   stage = 'begin' %stage of the computation, becomes 'done' when finished
   prevN %new data will be based on (quasi-)random vectors indexed by
   nextN %prevN + 1 to nextN
   timeUsed %time used so far
   nSamplesUsed %number of samples used so far
   confidInt = [-Inf Inf] %confidence interval for the solution
   costF %time required to compute function values
end

properties (Hidden)
   timeStart %starting time
end

methods (Abstract)
	updateData(obj, distribObj, fun_obj, decompType) %update the accumulated data
end
end
