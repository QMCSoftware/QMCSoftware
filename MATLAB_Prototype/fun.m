classdef fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties 
   f %function handle of integrand after transformation
   domain %domain of the function, §$\mcommentfont \cx$§
   domainType = 'box' %e.g., 'box', 'ball'
   dimension = 2 %dimension of the domain, §$\mcommentfont d$§
   distrib = struct('name','stdUniform') %e.g., 'uniform', 'Gaussian', 'Lebesgue'
   nominalValue = 0 %a nominal number, §$\mcommentfont c$§, such that §$\mcommentfont (c, \ldots, c) \in \cx$§
end

methods (Abstract)
   y = g(obj, xu, coordIndex)
   % xu = nodes, §\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix§
   % coordIndex = set of those coordinates in sequence needed, §\mcommentfont $\fu$§
   % y = §\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise§
end

methods
   function obj = transformVariable(obj,dstrObj)
   %This method performs the necessary variable transformation to put the
   %original function in the form required by the discreteDistributon
   %object
   for ii = 1:numel(obj)
      if strcmp(obj(ii).distrib.name,dstrObj.trueDistribution)
         obj(ii).f = @(xu, coordIndex) g(obj(ii), xu, coordIndex);
      elseif strcmp(obj(ii).distrib.name,'IIDZGaussian') && ... 
            strcmp(dstrObj.trueDistribution,'stdGaussian') %multiply by the likelihood ratio
         obj(ii).f = @(xu, coordIndex) g(obj(ii), xu*sqrt(obj.distrib.variance), ...
            coordIndex);
      end
   end
   end
end

end