classdef Penguin < Bird & ChocolateBar
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = Penguin(Name,Gender)
            this@Bird(Name,Gender,false);
            this@ChocolateBar('McVitie''s');
        end
        
    end
    
end