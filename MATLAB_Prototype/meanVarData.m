classdef meanVarData < accumData
% Accumulated data for IID calculations, stores the sample mean and
% variance of function values
properties
   muhat %sample mean
   sighat %sample standard deviation
   nSigma %number of samples used to compute the sample standard deviation
   nMu %number of samples used to compute the sample mean
end
methods 
   function obj = updateData(obj, distribObj, funObj)
      nf = numel(funObj);
      for ii = 1:nf
         tStart = tic; %time the function values
         dim = distribObj(ii).trueD.dimension;
         y = funObj(ii).f( ...
            genDistrib(disdistribObj(ii), obj.prevN(ii)+1, obj.prevN(ii)+obj.nextN(ii), ...
            obj.nextN(ii), 1:dim), 1:dim);
         obj.costF(ii) = toc(tStart); %to be used for multi-level methods
         if strcmp(obj.stage,'sigma')
            obj.sighat(ii) = std(y); %compute the sample standard deviation if required
         end
         obj.muhat(ii) = mean(y); %compute the sample mean
         obj.solution = sum(obj.muhat); %which also acts as our tentative solution
      end
   end
end
end
