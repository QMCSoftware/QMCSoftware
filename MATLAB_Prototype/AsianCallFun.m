classdef AsianCallFun < fun
% Specify and generate payoff values of an Asian Call option
properties
   volatility = 0.5
   S0 = 30;
   K = 25;
   BMmeasure = [];
   dimFac = 0;
end
methods
   function obj = AsianCallFun(BMmeasure)
      if nargin
         nBM = numel(BMmeasure);
         obj(1,nBM) = AsianCallFun;
         obj(1).BMmeasure = BMmeasure(1);
         obj(1).dimFac = 0;
         obj(1).dimension = BMmeasure(1).dimension;
         for ii = 2:nBM
            obj(ii).BMmeasure = BMmeasure(ii);
            obj(ii).dimFac = BMmeasure(ii).dimension/BMmeasure(ii-1).dimension;
            obj(ii).dimension = BMmeasure(ii).dimension;
         end
      end 
   end
   
   function y = g(obj, x, ~)
      SFine = obj.S0*exp((-obj.volatility^2/2)*obj.BMmeasure.measureData.timeVector + obj.volatility * x);
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