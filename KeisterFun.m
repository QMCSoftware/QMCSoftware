classdef KeisterFun < fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
methods
   function y = f(obj, x, coordIndex)
      %if the nominalValue = 0, this is efficient
      normx2 = sum(x.*x,2);
      if (numel(coordIndex) ~= obj.dimension) && (obj.nominalValue ~= 0)
         normx2 = normx2 + (obj.nominalValue.^2) * (obj.dimension - numel(coordIndex));
      end
      y = exp(-normx2) .* cos(sqrt(normx2));
   end
end
end