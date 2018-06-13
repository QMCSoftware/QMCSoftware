classdef meanVarData < accumData
%Accumulated data
properties
   muhat %sample mean
   sighat %sample variance
   lastN
   nextN
   nSigma
   nMu
   timeF
end
methods 
   function obj = updateData(obj, distribObj, funObj)
      nf = numel(funObj);
      for ii = 1:nf
         tStart = tic; %time the function values
         y = f(funObj(ii), ...
            genDistrib(distribObj, obj.lastN(ii)+1, obj.lastN(ii)+obj.nextN(ii), ...
            obj.nextN(ii), 1:funObj(ii).dimension, ii), 1:funObj(ii).dimension);
         obj.timeF(ii) = toc(tStart); %to be used for multi-level methods
         if strcmp(obj.stage,'sigma')
            obj.sighat(ii) = std(y);
         end
         obj.muhat(ii) = mean(y);
         obj.solution = sum(obj.muhat);
      end
   end
end
end
