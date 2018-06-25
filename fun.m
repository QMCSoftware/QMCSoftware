classdef (Abstract) fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties 
   domain = [0 0; 1 1] %domain of the function, §$\mcommentfont \cx$§
   domainType = 'box' %e.g., 'box', 'ball'
   dimension = 2 %dimension of the domain, §$\mcommentfont d$§
   distribType = 'uniform' %e.g., 'uniform', 'Gaussian'
   nominalValue = 0 %a nominal number, §$\mcommentfont c$§, such that §$\mcommentfont (c, \ldots, c) \in \cx$§
end

methods (Abstract)
	y = f(obj, xu, coordIndex)
   % xu = nodes, §\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix§
   % coordIndex = set of those coordinates in sequence needed, §\mcommentfont $\fu$§
   % y = §\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise§
end

end