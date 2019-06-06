classdef (Abstract) fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties 
   f %function handle of integrand after transformation
   dimension = 2
   nominalValue = 0 %a nominal number, §$\mcommentfont c$§, such that §$\mcommentfont (c, \ldots, c) \in \cx$§
end

methods (Abstract)
   y = g(obj, xu, coordIndex) %original function to be integrated
   % xu = nodes, §\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix§
   % coordIndex = set of those coordinates in sequence needed, §\mcommentfont $\fu$§
   % y = §\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise§
end

methods
   function obj = transformVariable(obj,msrObj,dstrObj)
   % This method performs the necessary variable transformation to put the
   % original function in the form required by the discreteDistributon
   % object starting from the original measure object
   %
   % msrObj = the measure object that defines the integral
   % dstrObj = the discrete distribution object that is sampled from
   for ii = 1:numel(obj)
      obj(ii).dimension = dstrObj(ii).trueD.dimension; %the function needs the dimension also
      if isequal(msrObj(ii),dstrObj(ii).trueD)
         obj(ii).f = @(xu, coordIndex) g(obj(ii), xu, coordIndex);
      elseif strcmp(msrObj(ii).measureName,'IIDZMeanGaussian') && ... 
            strcmp(dstrObj(ii).trueD.measureName,'stdGaussian') %multiply by the likelihood ratio
         obj(ii).f = @(xu, coordIndex) g(obj(ii), ...
            xu*sqrt(msrObj.measureData.variance), coordIndex);
      elseif strcmp(msrObj(ii).measureName,'BrownianMotion') && ... 
            strcmp(dstrObj(ii).trueD.measureName,'stdGaussian')
         timeDiff = diff([0 msrObj(ii).measureData.timeVector]);
         obj(ii).f = @(xu, coordIndex) g(obj(ii), ...
            cumsum(xu.*sqrt(timeDiff),2), coordIndex);
      else
         error('Variable transformation not performed')
      end
   end
   end
end

end