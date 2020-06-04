classdef Bird < Animal

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        canFly
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Bird(Name,Gender,CanFly)
            this@Animal(Name,Gender);
            this.canFly = CanFly;
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    
       
    methods        
        function output = getEggs(this,eggs)
            
            if(strcmp(this.getGender,'female'))
                output = ['You''ve just taken ' num2str(eggs) ...
                        ' eggs from ' this.name ' the bird.'];
            else
                output = ['You can''t get eggs from ' this.name ...
                           ' the bird, as it''s not female!'];
            end
                    
        end
        
    end
    
end