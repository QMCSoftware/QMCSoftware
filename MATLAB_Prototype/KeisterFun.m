classdef KeisterFun < fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
methods
   function y = g(obj, x, coordIndex)
      %if the nominalValue = 0, this is efficient
      normx2 = sum(x.*x,2);
      nCoordIndex = numel(coordIndex);
      if (nCoordIndex ~= obj.dimension) && (obj.nominalValue ~= 0)
         normx2 = normx2 + (obj.nominalValue.^2) * (obj.dimension - nCoordIndex);
      end
      y = (pi.^(nCoordIndex/2)).* cos(sqrt(normx2));
   end
   
   function obj = KeisterFun
      obj.distrib = struct('name','IIDZGaussian', ...
         'variance', 1/2);
   end
end
end