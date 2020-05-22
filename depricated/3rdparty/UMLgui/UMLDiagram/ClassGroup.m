classdef ClassGroup < handle
% CLASSGROUP is a generic group of ClassFile classes
%
% See also:
%   Tree ClassFile
%
% Ben Goddard 12/12/13
    
%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        classNameList
        classList
        nClasses
    end

%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------    
    
    methods
        
        function this = ClassGroup(nameList)

            % make list of class files
            this.nClasses = length(nameList);
            this.classNameList = nameList;
            this.classList = ClassFile(nameList{1});
            if(this.nClasses>1)
                for iClass = 2:this.nClasses
                    this.classList = [this.classList; ClassFile(nameList{iClass})];
                end
            end
            
            % assign class ids
            for iClass = 1:this.nClasses
                this.classList(iClass).id = iClass;
            end
            
        end
        
    end
    
end