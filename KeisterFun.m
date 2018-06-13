classdef KeisterFun < fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
properties
	domain = "unit cube" %domain of the discrete distribution, §$\mcommentfont \cx$§
	dimension = 2 %dimension of the domain, §$\mcommentfont d$§
   nominalValue = 0 %a nominal number, §$\mcommentfont c$§, such that §$\mcommentfont (c, \ldots, c) \in \cx$§
end
methods
   function y = f(obj, x, coordIndex)
      %since the nominalValue = 0, this is efficient
      normx2 = sum(x.*x,2);
      y = exp(-normx2) .* cos(sqrt(normx2));
   end
end
end