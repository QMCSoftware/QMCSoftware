classdef ClassFile < handle
% CLASSFILE class holds contents of a matlab class to pass to a tree
%
% See also:
%   ClassGroup Tree
%
% Ben Goddard 12/12/13

%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------
    
    properties (Access = public)
        name
        id
        
        % for constructing methods and properties text
        methodsFull
        props
        cutoffProp = {};
        addInputs = false;
        
        % family and level info
        superClasses
        parent
        parentId
        level
        extraParents = {};
        extraParentsIds
        
        % fonts
        nameFont    = 'Times-Bold';
        propsFont   = 'Times-Italic';
        methodsFont = 'Times-Roman';
        currentFont;
        altFont;
        changeFont;
        fontSize = 12;
        
        % strings for name, properties and methods
        nameString;
        propsStrings   = {};
        methodsStrings = {};
        currentStrings;
        centreString = false; % do we want centred text?
        
        % symbolds to denote method type
        publicSym    = '+';
        privateSym   = '-';
        protectedSym = '#';

        % list of abstract methods
        abstractMethods = [];
        
        % postscript output
        PS = '';
        
        % geometry
        PSX;
        PSY;
        X = 0;
        Y = 0;
        width  = 0;
        height = 0;
        xBorder = 4;
        yBorder = 4;
        boxX
        boxY
        boxW
        boxH
        
        % meta data including superclasses and methods
        metaData        
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------
    
    methods
        function this = ClassFile(name)
            if(nargin==0)
                
            else

                this.name = name;

                setMetaData(this);

                setNameString(this);

                % determine level
                setHierarchy(this);
            end
        end
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------
        
    methods
 
        %------------------------------------------------------------------
        % Get meta data
        %------------------------------------------------------------------
        
        function setMetaData(this)
            metaDataCmd = ['this.metaData  = ?' this.name ';'];
            eval(metaDataCmd);
        end
        

        %------------------------------------------------------------------
        % Get name, properties and methods strings
        %------------------------------------------------------------------
        
        function setNameString(this)
            this.nameString = {this.name};
        end
                
        %------------------------------------------------------------------
        % Get hierarchy information
        %------------------------------------------------------------------
        
        function setHierarchy(this)
            
            % get metadata for class
            md = [];
            metaDataCmd = ['md  = ?' this.name ';'];
            eval(metaDataCmd);
            
            % parents, i.e. direct superclasses
            parentsList = md.SuperclassList;
            
            % determine level recursively
            if(isempty(parentsList))
                this.level = 1;
                this.parent = 'handle';
            elseif(strcmp(parentsList(1).Name,'handle') ...
                    || strcmp(parentsList(1).Name,'value'))
                this.level = 1;
                this.parent = parentsList(1).Name;
            else    
                [parentLevel, this.parent] = getLevel(this.name,0,''); % recursion
                this.level = parentLevel + 1; % output is level of parent
            end
            
            % set extra parents when there are multiple superclasses
            nClassParents = length(parentsList);
            if(nClassParents>1)
                for jParent = 1:nClassParents
                    if(~strcmp(parentsList(jParent).Name,this.parent))
                        this.extraParents = [this.extraParents ; parentsList(jParent).Name];
                    end
                end
            end

            % auxilliary function to determine level by counting up the
            % tree and finding the deepest (primary) parent
            function [level,primaryParent] = getLevel(class,level,primaryParent)
                
                % get metadata and list of parents
                mdp = [];
                metaDataCmdP = ['mdp  = ?' class ';'];
                eval(metaDataCmdP);
                
                parents = mdp.SuperclassList;
                
                % run through all parents, recursively up the tree
                if(isempty(parents))
                    return;
                elseif(strcmp(parents(1).Name,'handle') ...
                        || strcmp(parents(1).Name,'value'))
                    return;
                else
                    % add one to the level for each level we go up the tree
                    level = level + 1;
                    
                    % recurse through parents and find deepest one
                    nParents = length(parents);
                    maxPLevel = 0;
                    for iParent = 1:nParents
                        thisparent = parents(iParent).Name;
                        pLevel = getLevel(thisparent,level);
                        if(pLevel>maxPLevel)
                            maxPLevel = pLevel;
                            primaryParent = thisparent;
                        end
                    end    
                    
                    % output deepest level
                    level = maxPLevel;
                    
                end
                
            end
            
        end

        %------------------------------------------------------------------
        % End of constructor methods
        %------------------------------------------------------------------

        
        %------------------------------------------------------------------
        % Determine strings to print for the properties
        %------------------------------------------------------------------

        function setPropsStrings(this)
            allProps = this.metaData.PropertyList;
            nProps = length(allProps);

            % only output those that are defined in this class, not in its
            % superclasses
            for iProp = 1:nProps
                if(strcmp(this.name, allProps(iProp).DefiningClass.Name))
                    this.propsStrings = [this.propsStrings, allProps(iProp).Name];
                end
            end
            
            % alphabetise
            [~,sortOrder] = sort(lower(this.propsStrings));
            this.propsStrings = this.propsStrings(sortOrder);
            
        end

        %------------------------------------------------------------------
        % Determine strings to print for the methods
        %------------------------------------------------------------------

        function setMethodsStrings(this)
            this.methodsStrings={};
            allMethods = this.metaData.MethodList;
            nMethods = length(allMethods);

            % position of the constructor method
            namePos = 0;
            
            for iMethod = 1:nMethods
                thisMethod = allMethods(iMethod);

                % only output those that are defined in this class, not in its
                % superclasses
                if(strcmp(this.name, thisMethod.DefiningClass.Name) ...
                       && ~thisMethod.Hidden );

                    % check if it's the constructor
                    if( strcmp(this.name, thisMethod.Name) )
                        namePos = iMethod;
                    end
                    
                    % add symbols to denote access
                    switch thisMethod.Access
                        case 'public'
                            visSym = this.publicSym;
                        case 'private'
                            visSym = this.privateSym;
                        case 'protected'
                            visSym = this.protectedSym;
                        otherwise
                            visSym = '?';
                    end

                    % add inputs if required
                    if(this.addInputs)
                        thisString = [visSym thisMethod.Name ' ('];          
                        inputs = thisMethod.InputNames;
                        nInputs = length(inputs);
                        
                        % don't print 'this' in all inputs
                        if(thisMethod.Static || iMethod == namePos)
                            firstInput = 1;
                        else
                            firstInput = 2;
                        end
                        
                        for iInput = firstInput:nInputs
                            thisInput = inputs{iInput};
                            if(iInput==firstInput)
                                thisString = [thisString ' ' thisInput];  %#ok
                            else
                                thisString = [thisString ', ' thisInput]; %#ok
                            end

                        end

                        thisString = [thisString ' )'];  %#ok
                        this.methodsStrings = [this.methodsStrings, thisString];
                        
                    else % if inputs not printed
                        this.methodsStrings = [this.methodsStrings, [visSym thisMethod.Name] ];
                    end
                    
                    % update list of abstract methods
                    if(thisMethod.Abstract)
                        this.abstractMethods = [this.abstractMethods; true];
                    else
                        this.abstractMethods = [this.abstractMethods; false];
                    end
                                       
                end
            end
            
            % remove constructor
            tempString = this.methodsStrings{namePos};
            this.methodsStrings(namePos) = [];
            this.abstractMethods(namePos) = [];
            
            % alphabetise
            [~,sortOrder] = sort(lower(this.methodsStrings));
            this.methodsStrings = this.methodsStrings(sortOrder);
            this.abstractMethods = this.abstractMethods(sortOrder);
            
            % add constructor at start if printing inputs
            if(this.addInputs)
                this.methodsStrings = [tempString, this.methodsStrings];
                this.abstractMethods = [false; this.abstractMethods];
            end
            
        end
        
                
        %------------------------------------------------------------------
        % Determine geometry
        %------------------------------------------------------------------
        
        function setWH(this)
            
            this.width = 0;
            
            nameWidth = getMaxStringWidth(this.nameString,this.nameFont,this.fontSize);
            this.width = max(this.width,nameWidth); 
            this.height = this.fontSize;
            
            propsWidth = getMaxStringWidth(this.propsStrings,this.propsFont,this.fontSize);
            this.width = max(this.width,propsWidth); 
            this.height = this.height + this.fontSize * length(this.propsStrings);

            methodsWidth = getMaxStringWidth(this.methodsStrings,this.methodsFont,this.fontSize);
            this.width = max(this.width,methodsWidth); 
            this.height = this.height + this.fontSize * length(this.methodsStrings);
            
            this.boxW = this.width  + 2*this.xBorder;
            this.boxH = this.height + 2*this.yBorder;
            
            % add space for line between name and first section
            if(~isempty(this.propsStrings) || ~isempty(this.methodsStrings))
                this.boxH = this.boxH + this.yBorder;
            end
            
            % add space for line between properties and methods sections,
            % if we print them both
            if(~isempty(this.propsStrings) && ~isempty(this.methodsStrings))
                this.boxH = this.boxH + this.yBorder;
            end
            
        end

        
        %------------------------------------------------------------------
        % Postscript functions
        %------------------------------------------------------------------
        
        %------------------------------------------------------------------
        % Main postscript constructor
        %------------------------------------------------------------------
        
        function setPS(this)
            this.PSX = this.X;
            this.PSY = this.Y;  % top left
            
            % postscript for name
            this.centreString = true;
            addClassPS(this);  
            this.centreString = false;
            
            % postscript for properties
            if(~isempty(this.propsStrings))
                this.PSY = this.PSY - this.yBorder;
            end
            addPropsPS(this); 

            % postscript for methods
            if(~isempty(this.methodsStrings))
                this.PSY = this.PSY - this.yBorder;
            end
            addMethodsPS(this);         
            
            % postscript for surrounding box
            addPSBox(this);
        end

        %------------------------------------------------------------------
        % Postscript for class name
        %------------------------------------------------------------------
        
        function addClassPS(this)                 
            this.currentFont = this.nameFont;
            
            % switch font if abstract
            if(this.metaData.Abstract)
                setEmphFont(this);
                this.currentFont = this.altFont;
            end

            % print font selection
            addPSFontChange(this);
            
            % print class name
            this.currentStrings = this.nameString;            
            this.changeFont = false(length(this.currentStrings));
            addPSStrings(this);
        end
        
        %------------------------------------------------------------------
        % Postscript for properties
        %------------------------------------------------------------------
        
        function addPropsPS(this)
            this.currentFont = this.propsFont;
            addPSFontChange(this);
            
            this.currentStrings = this.propsStrings;
            this.changeFont = false(length(this.currentStrings));
            addPSStrings(this);
        end
 
        %------------------------------------------------------------------
        % Postscript for methods
        %------------------------------------------------------------------
        
        function addMethodsPS(this)
            this.currentFont = this.methodsFont;
            addPSFontChange(this);
            
            this.currentStrings = this.methodsStrings;
            this.changeFont = this.abstractMethods;
            setEmphFont(this);
            
            addPSStrings(this);
        end
        
        %------------------------------------------------------------------
        % Postscript to change fonts
        %------------------------------------------------------------------

        function addPSFontChange(this)
            this.PS = [this.PS '/' this.currentFont ' findfont\n'];
            this.PS = [this.PS num2str(this.fontSize) ' scalefont setfont\n\n'];
        end
        
        %------------------------------------------------------------------
        % Determine emphasized version of font
        %------------------------------------------------------------------
        
        function setEmphFont(this)
            switch this.currentFont
                case 'Times-Bold'
                    this.altFont = 'Times-BoldItalic';
                case 'Times-Italic'
                    this.altFont = 'Times-Roman';
                otherwise
                    this.altFont = 'Times-Italic';
            end
        end

        %------------------------------------------------------------------
        % Print list of strings to postscript
        %------------------------------------------------------------------

        function addPSStrings(this)
            
            nStrings = length(this.currentStrings);
            printString = this.PS;

            for iString = 1:nStrings
                
                textString = this.currentStrings{iString};
                newFont = this.changeFont(iString);
                
                % determine X position depending on centering
                if(this.centreString)
                    stringWidth = getMaxStringWidth({textString},this.currentFont,this.fontSize);
                    freeSpace = this.width - stringWidth;
                    xPos = this.PSX + freeSpace/2;
                else
                    xPos = this.PSX;
                end
                
                % determine postscript to move and print text
                this.PSY = this.PSY - this.fontSize;    
                moveString = [num2str(xPos) ' ' num2str(this.PSY) ' moveto\n'];
                textString = ['(' textString ') show\n'];                 %#ok

                % change font of required
                if(newFont)
                    printString = [printString '\n/' this.altFont ' findfont\n']; %#ok
                    printString = [printString num2str(this.fontSize) ' scalefont setfont\n\n']; %#ok
                end
                
                % postscript output
                printString = [printString moveString textString]; %#ok
                
                % change font back if required
                if(newFont)
                    printString = [printString '/' this.currentFont ' findfont\n']; %#ok
                    printString = [printString num2str(this.fontSize) ' scalefont setfont\n\n']; %#ok
                end

            end
            
            % new line
            printString = [printString '\n'];
            
            % assign output
            this.PS = printString;
            
        end

        %------------------------------------------------------------------
        % Print box postscript
        %------------------------------------------------------------------

        function addPSBox(this)            

            % outline box
            this.boxX = this.X - this.xBorder;
            this.boxY = this.Y - this.fontSize/3 - this.boxH + this.yBorder;
   
            boxString = ['%% Box\n' num2str(this.boxX) ' ' num2str(this.boxY)  ...
                         ' ' num2str(this.boxW) ' ' num2str(this.boxH) ' rectstroke\n'];
            this.PS = [this.PS boxString];         

            % dividing line between name and first section
            if(~isempty(this.propsStrings) || ~isempty(this.methodsStrings))
                lineY = this.boxY + this.boxH - this.yBorder - this.fontSize;
                     
                lineMoveString = [num2str(this.boxX) ' ' num2str(lineY) ' moveto\n'];
                lineString = [ num2str(this.boxX + this.boxW) ' '  num2str(lineY) ' lineto\n'];

                this.PS = [this.PS lineMoveString lineString];
            end
                     
            % dividing line between first and second section
            if(~isempty(this.propsStrings) && ~isempty(this.methodsStrings))
                lineY = this.boxY + this.boxH - 2*this.yBorder ...
                        - this.fontSize*(length(this.propsStrings)+1);
                     
                lineMoveString = [num2str(this.boxX) ' ' num2str(lineY) ' moveto\n'];
                lineString = [ num2str(this.boxX + this.boxW) ' '  num2str(lineY) ' lineto\n'];

                this.PS = [this.PS lineMoveString lineString '\n'];
            end
             
        end
                       
    end
end