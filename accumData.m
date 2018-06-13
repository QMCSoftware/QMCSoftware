classdef (Abstract) accumData
%Accumulated data
properties
   solution = NaN %solution
   stage = 'begin'
   timeStart %starting time
   timeUsed %time used so far
   nSamplesUsed %number of samples used so far
   errorBound = [-Inf Inf] %error bound on the solution
end
methods (Abstract)
	updateData(obj, distribObj, fun_obj, decompType)
end
end
