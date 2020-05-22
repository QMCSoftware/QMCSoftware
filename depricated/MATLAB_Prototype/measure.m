classdef measure
% Specifies the components of a general measure used to define an
% integration problem or a sampling method
properties	
   dimension {mustBeInteger,mustBePositive} ...
      = 2 %dimension of the domain, §$\mcommentfont d$§
   domainCoord %domain coordinates for the measure, §$\mcommentfont \cx$§
   domainShape {mustBeMember(domainShape,{'', 'box', 'cube', 'unitCube'})} ...
      = '' %domain shape for the measure, §$\mcommentfont \cx$§
   measureName {mustBeMember(measureName,{'stdUniform', 'uniform', ...
      'stdGaussian', 'IIDZMeanGaussian', 'IIDGaussian', 'BrownianMotion', ...
      'Gaussian', 'Lesbesgue'})} ...
      = 'stdUniform' %name of the measure
   measureData %information required to specify the measure
end

methods
   
   %% This constructor allows us to set several properties with one command
   function obj = measure(varargin) %create measure with specified properties
      if nargin
         p = inputParser;
         addParameter(p,'dimension',2);
         addParameter(p,'domainCoord',[]);
         addParameter(p,'domainShape','');
         addParameter(p,'measureName','stdUniform');
         addParameter(p,'measureData',[]);
         parse(p,varargin{:})
         nObj = numel(p.Results.dimension);
         obj(1,nObj) =  measure;
         [obj.dimension] = p.Results.dimension{:};
         if ~any(strcmp('domainCoord',p.UsingDefaults))
            [obj.domainCoord] = p.Results.domainCoord{:};
         end
         if ~any(strcmp('domainShape',p.UsingDefaults))
            [obj.domainShape] = p.Results.domainShape{:};
         end
         if ~any(strcmp('measureName',p.UsingDefaults))
            [obj.measureName] = p.Results.measureName{:};
         end
         if ~any(strcmp('measureData',p.UsingDefaults))
            [obj.measureData] = p.Results.measureData{:};
         end
      end
   end
   
   %% Below we create functions to construct specific measures
   function obj = stdUniform(obj,varargin) %create standard uniform measure
      if nargin > 1
         p = inputParser;
         addParameter(p,'dimension',2); %only parse the required properties
         parse(p,varargin{:})
         nObj = numel(p.Results.dimension);
         obj(1,nObj) = measure;
         [obj.dimension] = p.Results.dimension{:};
      end
      [obj.measureName] = deal('stdUniform');
   end

   function obj = stdGaussian(obj,varargin) %create standard Gaussian measure
      if nargin > 1
         p = inputParser;
         addParameter(p,'dimension',2);
         parse(p,varargin{:})
         nObj = numel(p.Results.dimension);
         obj(1,nObj) = measure;
         [obj.dimension] = p.Results.dimension{:};
      end
      [obj.measureName] = deal('stdGaussian');
   end

   function obj = IIDZMeanGaussian(obj,varargin) %create standard Gaussian measure
      if nargin > 1
         p = inputParser;
         addParameter(p,'dimension',2);
         addParameter(p,'variance',1);
         parse(p,varargin{:})
         nObj = numel(p.Results.dimension);
         obj(1,nObj) = measure;
         [obj.dimension] = p.Results.dimension{:};
         [obj.measureData.variance] = p.Results.variance{:};
      end
      [obj.measureName] = deal('IIDZMeanGaussian');
   end
   
   function obj = BrownianMotion(obj,varargin) %create a discretized Brownian Motion measure
      p = inputParser;
      addParameter(p,'timeVector',{0.25:0.25:1});
      parse(p,varargin{:})
      nObj = numel(p.Results.timeVector);
      for ii = 1:nObj
         obj(ii).measureData.timeVector = p.Results.timeVector{ii};
         obj(ii).dimension = numel(obj(ii).measureData.timeVector);
      end
      [obj.measureName] = deal('BrownianMotion');
   end


end
end
