classdef KeisterFun < fun
% §\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
%
%  Examples
%
%  Example 1:
%
% >> kf = KeisterFun
%
%   kf =***
%     
%     domain: [2x2 double]
%     domainType: 'box'
%     dimension: 2
%     distribType: 'uniform'
%     nominalValue: 0    
%
%  >> kf.f([1,2; 3 4], [1 2])
% 
%   ans =
% 
%    -0.0042
%     0.0000
%
%
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