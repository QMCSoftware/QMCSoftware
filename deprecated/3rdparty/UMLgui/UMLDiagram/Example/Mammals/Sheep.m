classdef Sheep < Mammal

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        colour;
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Sheep(Name,Gender,Move,Colour)
            this@Mammal(Name,Gender,Move);
            this.colour = Colour;
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    

       
    methods        
        function output = getWool(this,skeins)
            output = ['You just shaved ' num2str(skeins) ' skeins of ' ...
                      this.colour ' wool from ' this.name ' the Sheep.'];
        end

    end    
    
end