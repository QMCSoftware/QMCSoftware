classdef Mammal < Animal

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        movementType
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Mammal(Name,Gender,Move)
            this@Animal(Name,Gender);
            this.movementType = Move;
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    
       
    methods        
        function output = getMilk(this,litres)

            if(strcmp(this.getGender,'female'))
                output = ['You''ve just taken ' num2str(litres) ...
                        ' litres of milk from ' this.name ' the mammal.'];
            else
                output = ['You can''t milk ' this.name ...
                           ' the mammal, as it''s not female!'];
            end
                    
        end
        
    end
    
end