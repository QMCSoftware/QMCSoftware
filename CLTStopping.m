classdef CLTStopping < stoppingCriterion
%§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties
   discDistAllowed = "IID" %which discrete distributions are supported
   decompTypeAllowed = "single" %which decomposition types are supported
   nInit = 1000 %initial sample size
   inflate = 1.2 %inflation factor
   alpha = 0.01;
end

properties (Dependent)
   quantile
end

methods
   
   function [obj, dataObj, distribObj] = ...
         stopYet(obj, dataObj, funObj, distribObj)
      if ~numel(dataObj)
         dataObj = meanVarData;
      end
      switch dataObj.stage
         case 'begin' %initialize
            dataObj.timeStart = tic;
            nf = numel(funObj);
            distribObj = initStreams(distribObj,nf);
            dataObj.lastN = zeros(1,nf); %initialize data object
            dataObj.nextN = repmat(obj.nInit,1,nf);
            dataObj.muhat = Inf(1,nf);
            dataObj.sighat = Inf(1,nf);
            dataObj.nSigma = obj.nInit;
            dataObj.timeF = zeros(1,nf);
            dataObj.stage = 'sigma';
         case 'sigma'
            dataObj.lastN = dataObj.nextN; %update place in the sequence
            tempA = sqrt(dataObj.timeF);
            tempB = sum(tempA .* dataObj.sighat);
            nM = ceil((tempB*(obj.quantile*obj.inflate ...
               /max(obj.absTol,dataObj.solution*obj.relTol))^2) ...
               * (dataObj.sighat./sqrt(dataObj.timeF)));
            dataObj.nMu = max(dataObj.nextN,nM);
            dataObj.nextN = dataObj.nMu + dataObj.lastN;
            dataObj.stage = 'mu';
         case 'mu'
            dataObj.solution = sum(dataObj.muhat);
            dataObj.nSamplesUsed = dataObj.nextN;
            errBar = (obj.quantile * obj.inflate) * ...
               sqrt(sum(dataObj.sighat.^2/dataObj.nMu));
            dataObj.errorBound = dataObj.solution + errBar*[-1 1];
            dataObj.stage = 'done';
      end
      dataObj.timeUsed = toc(dataObj.timeStart);
   end
 
   function value = get.quantile(obj)
      value = -norminv(obj.alpha/2);
   end
   
 end
end

