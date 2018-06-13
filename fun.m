classdef (Abstract) fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties (Abstract)
   domain %domain of the discrete distribution, §$\mcommentfont \cx$§
   dimension %dimension of the domain, §$\mcommentfont d$§
end
methods (Abstract)
	y = f(obj, x, coordIndex)
    % xu = nodes, §\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix§
    % coordIndex = set of those coordinates in sequence needed, §\mcommentfont $\fu$§
    % y = §\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise§
end
end