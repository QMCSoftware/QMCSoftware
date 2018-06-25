classdef AsianCallFun < fun
% §\mcommentfont Specify and generate payoff values of an Asian Call option$§
properties
   volatility = 0.5
   S0 = 30;
   K = 25;
   T = 1;
   A
   tVec
   dimFac = 1 
end
methods
   function obj = AsianCallFun(dimFac)
      if nargin
         dimVec = cumprod(dimFac);
         nf = numel(dimVec);
         obj(1,nf) = AsianCallFun;
         obj(1).dimFac = 0;      
         for ii = 1:nf
            d = dimVec(ii);
            if ii > 1
               obj(ii).dimFac = dimFac(ii-1);
            end
            obj(ii).dimension = d;
            tvec = (1:d)*(obj(ii).T/d);
            obj(ii).tVec = tvec;
            CovMat = min(tvec',tvec);
            [eigVec,eigVal] = eig(CovMat,'vector');
            obj(ii).A = sqrt(eigVal(end:-1:1)) .* eigVec(:,end:-1:1)';
         end
      end 
   end
   
   function y = f(obj, x, coordIndex)
      %since the nominalValue = 0, this is efficient
      BM = x * obj.A;
      SFine = obj.S0*exp((-obj.volatility^2/2)*obj.tVec + obj.volatility * BM);
      AvgFine = ((obj.S0/2) + sum(SFine(:,1:obj.dimension-1),2) + ...
         SFine(:,obj.dimension)/2)/obj.dimension;
      y = max(AvgFine - obj.K,0);
      if obj.dimFac > 0
         SCoarse = SFine(:,obj.dimFac:obj.dimFac:end);
         dCoarse = obj.dimension/obj.dimFac;
         AvgCoarse = ((obj.S0/2) + sum(SCoarse(:,1:dCoarse-1),2) + ...
            SCoarse(:,dCoarse)/2)/dCoarse;
         y = y - max(AvgCoarse - obj.K,0);
      end
   end
end
end