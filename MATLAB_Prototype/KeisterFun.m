classdef KeisterFun < fun
% §\mcommentfont Specify and generate values $f(\vx) = \pi^{d/2} \cos(\lVert \vx \rVert)$ for $\vx \in \reals^d$§
% The standard example integrates the Keister function with respect to an IID Gaussian distribution with variance 1/2
% B. D. Keister, Multidimensional Quadrature Algorithms, §\mcommentfont \emph{Computers in Physics}, \textbf{10}, pp.\ 119-122, 1996.§
methods
   function y = g(obj, x, coordIndex)
      %if the nominalValue = 0, this is efficient
      normx2 = sum(x.*x,2);
      nCoordIndex = numel(coordIndex);
      if (nCoordIndex ~= obj.dimension) && (obj.nominalValue ~= 0)
         normx2 = normx2 + ...
            (obj.nominalValue.^2) * (obj.dimension - nCoordIndex);
      end
      y = (pi.^(nCoordIndex/2)).* cos(sqrt(normx2));
   end
end
end