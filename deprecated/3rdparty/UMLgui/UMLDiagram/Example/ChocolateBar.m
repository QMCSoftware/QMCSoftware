classdef ChocolateBar < handle

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        brand
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = ChocolateBar(Brand)
            if(nargin==0)
                this.brand = [];
            else
                this.brand = Brand;
            end
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    
       
    methods        
        function output = eat(this)
            output = ['You''ve just eaten a chocolate bar from ' ...
                       this.brand '. Nom!'];
        end
        
    end
    
end