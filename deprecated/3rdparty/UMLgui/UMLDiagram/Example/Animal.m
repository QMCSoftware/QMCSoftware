classdef Animal < handle

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        name
    end

    properties (Access = private)
        gender
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Animal(Name,Gender)
            this.name = Name;
            this.gender = Gender;
        end
        
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    
       
    methods        
        function description = describe(this)
            description = [this.name ' is a ' this.gender ' animal'];
        end

    end
    
    methods (Access = protected)
        function thisGender = getGender(this)
            thisGender = this.gender;
        end        
    end
    
    
end