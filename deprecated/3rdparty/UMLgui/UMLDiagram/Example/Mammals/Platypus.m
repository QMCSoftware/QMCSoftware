classdef Platypus < Mammal

%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Platypus(Name,Gender,Move)
            this@Mammal(Name,Gender,Move);
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    

       
    methods        
        function output = injectVenom(this,target)
            output = [this.name ' just injected ' target ' with venom!'];
        end

    end    
    
end