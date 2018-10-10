classdef CLTStopping < stoppingCriterion
% Stopping criterion based on the Central Limit Theorem
properties
   discDistAllowed = 'IIDDistribution' %which discrete distributions are supported
   decompTypeAllowed = {'single', 'multi'} %which decomposition types are supported
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
         dataObj = meanVarData; %create a new accumulated data object
      end
      switch dataObj.stage
         case 'begin' %initialize
            dataObj.timeStart = tic; %keep track of time
            if ~any(strcmp(obj.discDistAllowed,class(distribObj)))
               error('Stopping criterion not compatible with sampling distribution')
            end
            nf = numel(funObj); %number of functions whose integrals add up to the solution
            distribObj = initStreams(distribObj,nf); %need an IID stream for each function
            dataObj.prevN = zeros(1,nf); %initialize data object
            dataObj.nextN = repmat(obj.nInit,1,nf);
            dataObj.muhat = Inf(1,nf);
            dataObj.sighat = Inf(1,nf);
            dataObj.nSigma = obj.nInit; %use initial samples to estimate standard deviation
            dataObj.costF = zeros(1,nf);
            dataObj.stage = 'sigma'; %compute standard deviation next
         case 'sigma'
            dataObj.prevN = dataObj.nextN; %update place in the sequence
            tempA = sqrt(dataObj.costF); %use cost of function values to decide how to allocate
            tempB = sum(tempA .* dataObj.sighat); %samples for computation of the mean
            nM = ceil((tempB*(obj.quantile*obj.inflate ...
               /max(obj.absTol,dataObj.solution*obj.relTol))^2) ...
               * (dataObj.sighat./sqrt(dataObj.costF)));
            dataObj.nMu = min(max(dataObj.nextN,nM),obj.nMax - dataObj.prevN);
            dataObj.nextN = dataObj.nMu + dataObj.prevN;
            dataObj.stage = 'mu'; %compute sample mean next
         case 'mu'
            dataObj.solution = sum(dataObj.muhat);
            dataObj.nSamplesUsed = dataObj.nextN;
            errBar = (obj.quantile * obj.inflate) * ...
               sqrt(sum(dataObj.sighat.^2/dataObj.nMu));
            dataObj.errorBound = dataObj.solution + errBar*[-1 1];
            dataObj.stage = 'done'; %finished with computation
      end
      dataObj.timeUsed = toc(dataObj.timeStart);
   end
 
   function value = get.quantile(obj)
      value = -norminv(obj.alpha/2);
   end
   
 end
end

