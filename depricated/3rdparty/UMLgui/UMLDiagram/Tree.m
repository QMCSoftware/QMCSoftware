classdef Tree < ClassGroup
% TREE class creates and plots a tree of ClassFile classes 
%
% See also:
%   ClassGroup ClassFile
%
% Ben Goddard 12/12/13
  
%--------------------------------------------------------------------------
% PROPERTIES
%--------------------------------------------------------------------------    
    
    properties (Access = public)
        
        % level information
        levelLists
        levelIds
        nInLevels
        nLevels
                
        % family information
        parentIds
        
        firstKidIds
        lastKidIds
        lSibIds
        rSibIds
        kidLists
        prevInLevel
        nKids
        
        % for fancy plotting
        Xprelim
        Xmod
        ancestors
        threads
        changes
        shifts
        
        % geometry
        levelWidths
        levelHeights
        cumHeight
        UMLWidth
        UMLHeight
        Xgap    = 50;
        Ygap    = 150;
        border  = 0;
        
        % split arrows in compact plotting
        splitLevels = false;
        
        % options for what to include
        doProps   = true;
        doMethods = true;
        doMethodsInputs = true;
        
        % font information
        nameFont    = 'Times-Bold';
        propsFont   = 'Times-Italic';
        methodsFont = 'Times-Roman';
        fontSize = 12;
        triangleSize = 12;
        
        % flag to determine if the classes have been sorted into levels
        sorted = false;
        
        % output file
        saveFile = [];
        
        % plot title
        title = [];
        
        % options stored in this when called externally
        options = [];
        
        % flag to determine if we use fancy (true) or compact (false) plots
        fancy = true;
        
    end
    
%--------------------------------------------------------------------------
% CONSTRUCTOR
%--------------------------------------------------------------------------
   
    methods
        function this = Tree(nameList,opts)
            
            % make a tree
            this@ClassGroup(nameList);
            
            % allocate any provided options
            if(nargin==2)
                this.options = opts;
                setOpts(this);
            end
            
            % determine levels of each class in the UML
            setLevelInfo(this);
            
            % set plotting type
            if(this.fancy)
                setFamilyIds(this);
            else
                setParentIds(this);
            end

            
            if(~this.sorted)
                
                % resort classes so they're in level order
                % only do this on the first run through
                % as then sorted flag is set to true
                this = resort(this);
                
            else % second run through with sorted classes
                
                % set fonts
                setFonts(this);
                
                % set which parts of properties and methods info
                % is outputted
                setPropsAndMethods(this);
                
                % set widths and heights of classes
                setWHs(this);
                
                % determine level geometries
                setLevelGeoms(this)
                
                % set postscript positions in each class
                if(this.fancy)
                    setPSPosFancy(this);
                else
                    setPSPos(this);
                end
                
                % make postscript (and pdf) output
                makePS(this);

            end
            
        end
    end

%--------------------------------------------------------------------------
% METHODS
%--------------------------------------------------------------------------    
    
    methods
        
        %------------------------------------------------------------------
        % Set options from options input
        %------------------------------------------------------------------
        
        function setOpts(this)
            
            % list of possible options
            fieldList = {'sorted', ...
                         'Xgap','Ygap', ...
                         'doProps','doMethods','doMethodsInputs', ...
                         'nameFont','propsFont','methodsFont','fontSize', ...
                         'saveFile','title','fancy'};

            % set options for those that exist in the input
            tempOpts = this.options;
            for iField = 1:length(fieldList)
                field = fieldList{iField};
                if(isfield(tempOpts,field))
                    this.(field) = tempOpts.(field);
                end
            end

        end
        
        %------------------------------------------------------------------
        % Determine and set parent or family information for compact 
        % plotting.  Note this only requires parent id and number of
        % children
        %------------------------------------------------------------------
                
        function setParentIds(this)
            parents = zeros(this.nClasses,1);
            kidsCount = zeros(this.nClasses,1);
        
            % determine parent id of each class
            for iClass = 1:this.nClasses
                parent = this.classList(iClass).parent;
                parentId = find(strcmp(this.classNameList,parent));
                
                % add children to the parent
                if(isempty(parentId))
                    parentId = 0;
                else
                    kidsCount(parentId)=kidsCount(parentId)+1;
                end
                
                this.classList(iClass).parentId = parentId;
                parents(iClass) = parentId;
                                
            end
            
            % these are vector of length nClasses
            this.parentIds = parents;
            this.nKids = kidsCount;
            
            % add parents that live higher up the tree to each class
            setExtraParentsIDs(this);
                        
        end
        
        %------------------------------------------------------------------
        % Add extra parents (i.e. for those classes with multiple
        % superclasses) that live higher up the tree
        %------------------------------------------------------------------
        
        function setExtraParentsIDs(this)
            
            for iClass = 1:this.nClasses
                extraParents = this.classList(iClass).extraParents;

                % set ids of extra parents (rather than class names)
                if(~isempty(extraParents))
                    nEP = length(extraParents);
                    EPIds = zeros(nEP,1);
                    for iEP = 1:nEP
                        EPIds(iEP) = find(strcmp(this.classNameList,extraParents{iEP}));
                    end
                    this.classList(iClass).extraParentsIds = EPIds;
                end

            end

        end
        
        %------------------------------------------------------------------
        % Determine and set parent or family information for fancy
        % plotting.  Note this only requires a lot more information as the
        % plotting algorithm is more complicated.
        %------------------------------------------------------------------

        function setFamilyIds(this)
            % preallocate
            parents = zeros(this.nClasses + 1,1);
            kidsCount = zeros(this.nClasses + 1,1);
            kids = cell(this.nClasses + 1,1);
            firstKids = zeros(this.nClasses + 1,1);
            lastKids = zeros(this.nClasses + 1,1);
            lSibs = zeros(this.nClasses + 1,1);
            rSibs  = zeros(this.nClasses + 1,1);
            previous = zeros(this.nClasses + 1,1);
            this.Xprelim = zeros(this.nClasses+1,1);
            this.Xmod = zeros(this.nClasses+1,1);

            % set final class, which is a dummy one to use for the root
            % of the tree
            this.classList(this.nClasses + 1) = ClassFile();
            this.classList(this.nClasses + 1).id = this.nClasses + 1;
            this.classList(this.nClasses + 1).level = 0;
            
            % allocate parents, kids and previous class in the level                        
            for iClass = 1:this.nClasses
                parent = this.classList(iClass).parent;
                
                parentId = find(strcmp(this.classNameList,parent));
                
                if(isempty(parentId))
                    parentId = 0;
                else
                    kidsCount(parentId)=kidsCount(parentId)+1;
                    kids{parentId} = [kids{parentId}; iClass];
                end
                
                this.classList(iClass).parentId = parentId;
                parents(iClass) = parentId;
                
                if(iClass>1)
                    if(this.levelIds(iClass-1) == this.levelIds(iClass))
                        previous(iClass) = iClass - 1;
                    end
                end
            end

            % determine ordering of children and siblings
            for iClass = 1:this.nClasses+1
                if(~isempty(kids{iClass}))
                    firstKids(iClass) = kids{iClass}(1);
                    lastKids(iClass) = kids{iClass}(end);
                end
                
                if(parents(iClass) ~= 0)
                    siblings = kids{parents(iClass)};
                else
                    siblings = this.levelLists{1};
                end
                    
                thisPos = find(siblings == iClass);

                if(thisPos > 1)
                    lSibs(iClass) = siblings(thisPos - 1);
                end

                if(thisPos < length(siblings))
                    rSibs(iClass) = siblings(thisPos + 1);
                end 
            end
            
            % dummy info for root
            kids{this.nClasses+1} = this.levelLists{1};
            firstKids(this.nClasses+1) = this.levelLists{1}(1);
            lastKids(this.nClasses+1) = this.levelLists{1}(end);
            kidsCount(this.nClasses+1) = length(this.levelLists{1});
            lSibs(this.nClasses+1) = 0;
        	rSibs(this.nClasses+1) = 0;

            % allocate data
            this.parentIds = parents;
            this.nKids = kidsCount;
            this.kidLists = kids;
            this.firstKidIds = firstKids;
            this.lastKidIds = lastKids;
            this.lSibIds = lSibs;
            this.rSibIds = rSibs;
            this.prevInLevel = previous;
            
            % add parents that live higher up the tree to each class
            setExtraParentsIDs(this);

        end
        
        %------------------------------------------------------------------
        % Determine and set level information
        %------------------------------------------------------------------
        
        function setLevelInfo(this)
            
            % find number of different levels in the tree
            this.nLevels = 0;            
            for iClass = 1:this.nClasses
                this.nLevels = max(this.nLevels,this.classList(iClass).level);    
            end
            
            % preallocate
            this.levelLists = cell(this.nLevels,1);
            this.levelIds = zeros(this.nClasses,1);
            this.nInLevels = zeros(this.nLevels,1);
            
            % allocate each class to its level list and set its levelId
            for iClass = 1:this.nClasses
                level = this.classList(iClass).level;
                if( isempty(this.levelLists{level}) )
                    this.levelLists{level} = iClass;
                else
                    this.levelLists{level} = [this.levelLists{level}; iClass];
                end
                this.levelIds(iClass) = level;
            end
            
            % find number of classes in each level
            for iLevel = 1:this.nLevels
               this.nInLevels(iLevel) = sum(this.levelIds==iLevel);
            end
                               
        end

        %------------------------------------------------------------------
        % Re-sort classes in tree according to level and parents
        %------------------------------------------------------------------

        function sortedThis = resort(this)

            % preallocate
            newPos = zeros(this.nClasses,1);

            % new postiion for level 1
            newPos(this.levelLists{1}) = 1:this.nInLevels(1);

            % keep track of the total offset for previous levels
            cumNInLevels = cumsum(this.nInLevels);

            % sort each level so that classes with the same superclasses
            % (parents) are next to each other
            for iLevel = 2:this.nLevels
                levelMask = (this.levelIds==iLevel);
                levelList = this.levelLists{iLevel};
                nInLevel = this.nInLevels(iLevel);

                levelParents = this.parentIds(levelMask);
                newLevelParents = newPos(levelParents);

                [~,sortMask] = sort(newLevelParents);
                newPos(levelList(sortMask)) = (1:nInLevel) + cumNInLevels(iLevel-1);

            end
            newNameList(newPos) = this.classNameList;
            
            % set the sorted flag to true and reinitialise the tree
            tempOpts = this.options;
            tempOpts.sorted = true;
            sortedThis = Tree(newNameList,tempOpts);
           
        end
                
        %------------------------------------------------------------------
        % Set fonts, text, widths and heights in classes
        %------------------------------------------------------------------

        function setFonts(this)
            for iClass = 1:this.nClasses
                this.classList(iClass).nameFont = this.nameFont;
                this.classList(iClass).propsFont = this.propsFont;
                this.classList(iClass).methodsFont = this.methodsFont;
                this.classList(iClass).fontSize = this.fontSize;
            end
        end
                   
        function setPropsAndMethods(this)
            for iClass = 1:this.nClasses
                if(this.doProps)
                    this.classList(iClass).setPropsStrings;
                end
                if(this.doMethods)
                    this.classList(iClass).addInputs = this.doMethodsInputs;
                    this.classList(iClass).setMethodsStrings;
                end     
            end
        end
        
        % can't do this until we've determined what information we want to
        % include, so it can't be in the class constructor
        function setWHs(this)
            for iClass = 1:this.nClasses
                this.classList(iClass).setWH;
            end
        end
        
        
        %------------------------------------------------------------------
        % Determine and set level widths and heights
        %------------------------------------------------------------------
                
        function setLevelGeoms(this)
  
            % preallocate
            this.levelWidths = zeros(this.nLevels,1);
            this.levelHeights = zeros(this.nLevels,1);
            
            % determine level widths and heights from classes
            for iLevel = 1:this.nLevels
                levelClasses = this.levelLists{iLevel};
                nClasses = length(levelClasses);
                for iClass = 1:nClasses
                    this.levelWidths(iLevel) = this.levelWidths(iLevel) + this.classList(levelClasses(iClass)).boxW;
                    this.levelHeights(iLevel) = max( this.levelHeights(iLevel), this.classList(levelClasses(iClass)).boxH );
                end
                this.levelWidths(iLevel) = this.levelWidths(iLevel) + (nClasses-1)*this.Xgap;
            end     
            
            % determine cumulative height to top of box
            this.cumHeight(this.nLevels) = this.levelHeights(this.nLevels);
            for iLevel = this.nLevels-1:-1:1
                this.cumHeight(iLevel) = this.cumHeight(iLevel+1) + this.Ygap + this.triangleSize + this.levelHeights(iLevel);
            end
            
            % determine total width and height
            this.UMLHeight = this.cumHeight(1);
            [this.UMLWidth,~] = max(this.levelWidths);
      
            % add extra padding to edges
            addBorders(this);
            
        end
        
        %------------------------------------------------------------------
        % Add borders to plot
        %------------------------------------------------------------------
        
        function addBorders(this)
            
            % paper height and 1 inch border in pts
            a4Height = 852;
            inchBorder = 72;
            
            if(this.UMLHeight/this.UMLWidth>sqrt(2))  % portrait, border determined by height
                this.border = ceil(this.UMLHeight*inchBorder/a4Height);
            else % landscape
                this.border = ceil(this.UMLWidth*inchBorder/a4Height);
            end

            % adjust geometries
            this.UMLWidth = this.UMLWidth + 2*this.border;
            this.UMLHeight = this.UMLHeight + 2*this.border;
            this.cumHeight =  this.cumHeight + this.border;

        end

        %------------------------------------------------------------------
        % Get the positions of each class and store in the class,
        % for compact plotting
        %------------------------------------------------------------------
        
        function setPSPos(this)
            
            cumWidth = this.border*ones(this.nLevels,1);

            for iLevel = 1:this.nLevels
                levelList = this.levelLists{iLevel};
                nClasses = length(levelList);
                thisClass = this.classList(levelList(1));

                % set position for first class in level
                thisClass.X = cumWidth(iLevel) + thisClass.xBorder;
                thisClass.Y = this.cumHeight(iLevel) + thisClass.yBorder;

                % set all other positions iteratively
                for iClass = 2:nClasses
                    thisClass = this.classList(levelList(iClass));
                    cumWidth(iLevel) = cumWidth(iLevel) + this.Xgap + this.classList(levelList(iClass-1)).boxW;
                    thisClass.X = cumWidth(iLevel) + thisClass.xBorder;
                    thisClass.Y = this.cumHeight(iLevel) + thisClass.yBorder;
                end

                    
            end
            
        end

        %------------------------------------------------------------------
        % Get the positions of each class and store in the class,
        % for fancy plotting
        %------------------------------------------------------------------

        function setPSPosFancy(this)
            
            % preallocate
            this.Xmod = zeros(this.nClasses+1,1);
            this.Xprelim = zeros(this.nClasses+1,1);
            this.ancestors = (1:(this.nClasses+1)).';
            this.threads = zeros(this.nClasses+1,1);
            this.changes = zeros(this.nClasses+1,1);
            this.shifts = zeros(this.nClasses+1,1);
            
            % do first walk of tree, note the inclusion of a root class
            firstWalk(this,this.nClasses+1);
            % do second walk of tree, note the inclusion of a root class
            secondWalk(this,this.nClasses+1,0);
            
            % determine max and min X and Y
            maxX = 0;
            minX = 0;
            maxY = 0;
            minY = 0;
            
            for iClass = 1:this.nClasses
                thisClass = this.classList(iClass);
                thisClass.X = thisClass.X - thisClass.width/2;
                maxX = max(maxX,thisClass.X + thisClass.boxW);
                minX = min(minX,thisClass.X);
                maxY = max(maxY,thisClass.Y);
                minY = min(minY,thisClass.Y - thisClass.boxH);
            end
            
            % determine width and add borders
            this.UMLWidth = maxX - minX;                    
            addBorders(this);
            
            
            % make all X and Y positive
            for iClass = 1:this.nClasses
                thisClass = this.classList(iClass);
                thisClass.X = thisClass.X - minX + this.border;
                thisClass.Y = thisClass.Y - minY + this.border;
            end
            
        end
        
        %------------------------------------------------------------------
        % Make eps and pdf output
        %------------------------------------------------------------------
        
        function makePS(this)
            
            % get time and date info
            time = clock;
            dateString = datestr(now);
            dateFont = 'Times-Roman';
            dateSize = this.fontSize;
            dateWidth = getMaxStringWidth({dateString},dateFont,dateSize);
            nowStr =  [num2str(time(1)),'_',... % Returns year as char
                       num2str(time(2)),'_',... % Returns month as char
                       num2str(time(3)),'_',... % Returns day as char
                       num2str(time(4)),'_',... % returns hour as char
                       num2str(time(5)),'_',... % Returns minute as char
                       num2str(time(6))];       % Returns seconds as char
                   
                   
            % determine save file
            if(isempty(this.saveFile))
                thisFolder = fileparts(mfilename('fullpath'));
                saveFolder = [thisFolder filesep 'Diagrams'];
                if(~(exist(saveFolder,'dir')==7))
                        mkdir(saveFolder);
                end
                this.saveFile = [saveFolder filesep 'UMLdiagram-' nowStr];
            else
                [saveFolder, saveFileNew, saveExt] = fileparts(this.saveFile);
                if(isempty(saveExt))
                    if(~(exist(this.saveFile,'dir')==7))
                        mkdir(this.saveFile);
                    end
                    if(strcmp(this.saveFile(end),filesep))
                        this.saveFile = [this.saveFile 'UMLdiagram-' nowStr];
                    else
                        this.saveFile = [this.saveFile filesep 'UMLdiagram-' nowStr];
                    end
                else
                    this.saveFile = [saveFolder filesep saveFileNew];
                end
                
            end
                
            saveFileEPS = [this.saveFile '.eps'];
            saveFilePDF = [this.saveFile '.pdf'];

            % print start of eps file
            fid = fopen(saveFileEPS,'w');
            fprintf(fid, '%%!PS-Adobe-3.0 EPSF-3.0\n');
            fprintf(fid, ['%%%%BoundingBox: 0 0 ' num2str(this.UMLWidth) ' ' num2str(this.UMLHeight) '\n']);
            fprintf(fid, '\n');
            
            commentLine = '%%---------------------------------------------%%\n';
            
            % print postscript for each class
            for iClass = 1:this.nClasses
                % determine postscript output from inside the class
                this.classList(iClass).setPS;
                
                % and print it to the file
                fprintf(fid, ['\n' commentLine]);
                fprintf(fid, ['%% Begin ' this.classList(iClass).name '\n']);
                fprintf(fid, [commentLine '\n']);                
                fprintf(fid, this.classList(iClass).PS );
                
                % get arrow position and print to file
                if(this.parentIds(iClass)>0 && this.parentIds(iClass)<this.nClasses)
                    if(this.fancy)
                        [arrowX,arrowY] = getArrowFancy(this,iClass);
                    else
                        [arrowX,arrowY] = getArrow(this,iClass);
                    end
                    fprintf(fid,  this.makeArrowPS(arrowX,arrowY));
                end
                
                % add straight arrows for extra parents (this needs work)
                if(~isempty(this.classList(iClass).extraParentsIds))
                    EPIds = this.classList(iClass).extraParentsIds;
                    for iEP = 1:length(EPIds)
                        [arrowX,arrowY] = getArrowStraight(this,iClass,EPIds(iEP));
                        fprintf(fid,  this.makeArrowPS(arrowX,arrowY));
                    end
                end
                
                % add comment block
                fprintf(fid, ['\n' commentLine]);
                fprintf(fid, ['%% End ' this.classList(iClass).name '\n']);
                fprintf(fid, [commentLine '\n']);

            end
            
            % add date
            fprintf(fid, ['\n/' dateFont ' findfont\n']);
            fprintf(fid, [num2str(dateSize) ' scalefont setfont\n\n']);
            fprintf(fid, [num2str(this.UMLWidth - this.border - 1.1*dateWidth) ' ' num2str(this.UMLHeight-this.border + dateSize) ' moveto\n']);
            fprintf(fid, ['(' datestr(now) ') show\n']);
            
            % add title
            if(~isempty(this.title))
                titleWidth = getMaxStringWidth({this.title},dateFont,dateSize);
                fprintf(fid, [num2str((this.UMLWidth - titleWidth)/2) ' ' num2str(this.UMLHeight-this.border + dateSize) ' moveto\n']);
                fprintf(fid, ['(' this.title ') show\n']);
            end

            % finish postscript file
            fprintf(fid, '\nshowpage');
            fclose(fid);
            
            % Set ghostscript options to convert to pdf
            GSopts = [' -q -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile="' saveFilePDF '" -f "' saveFileEPS '"'];

            % Get gs command depending on OS
            switch computer
                    case {'MAC','MACI','MACI64'}			
                        gs= '/usr/local/bin/gs';
                    case {'PCWIN'}
                        gs= 'gswin32c.exe';
                    case {'PCWIN64'}
                        gs = 'C:\gswin64c.exe';
                    otherwise
                        gs= 'gs';
            end

            % Use gs to convert to pdf
            eps2pdfCmd=[ gs GSopts];
            system(eps2pdfCmd);
            this.saveFile = saveFilePDF;
            
        end

        %------------------------------------------------------------------
        % Get postscript for arrows for compact plotting
        %------------------------------------------------------------------
        
        function arrowPS = makeArrowPS(this,X,Y)
            arrowPS = '%% Arrow\n';
            arrowPS = [arrowPS num2str(X(1)) ' ' num2str(Y(1)) ' moveto\n'];
            for iPt = 1:(length(X)-1)
                arrowPS = [arrowPS num2str(X(iPt)) ' ' num2str(Y(iPt)) ' lineto\n'];  %#ok
            end
            arrowPS = [arrowPS num2str(X(end)) ' ' num2str(Y(end) - this.triangleSize) ' lineto\n'];
            
            arrowPS = [arrowPS num2str(X(end)) ' ' num2str(Y(end)) ' moveto\n'];
            arrowPS = [arrowPS num2str(X(end) - this.triangleSize) ' ' num2str(Y(end) - this.triangleSize) ' lineto\n'];
            arrowPS = [arrowPS num2str(X(end) + this.triangleSize) ' ' num2str(Y(end) - this.triangleSize) ' lineto\n'];
            arrowPS = [arrowPS num2str(X(end)) ' ' num2str(Y(end)) ' lineto\n'];
            arrowPS = [arrowPS '\nstroke\n'];
 
        end

        %------------------------------------------------------------------
        % Get postscript for arrows for fancy plotting
        %------------------------------------------------------------------

        function [arrowX,arrowY] =getArrowFancy(this,iClass)
            thisClass = this.classList(iClass);
            parentId = thisClass.parentId;
                        
            if(parentId>0)
                thisBoxMidX = thisClass.boxX + thisClass.boxW/2;
                thisBoxMidY = thisClass.boxY + thisClass.boxH;
                parentClass = this.classList(parentId);
                pEndX = parentClass.boxX + parentClass.boxW/2;
                pEndY = parentClass.boxY;
                midY = thisBoxMidY +  this.Ygap/2;
                
                arrowX = [thisBoxMidX,thisBoxMidX,pEndX,pEndX];
                arrowY = [thisBoxMidY,midY,midY,pEndY];
            else
                arrowX = [NaN;NaN];
                arrowY = [NaN;NaN];
            end
        end
        
        %------------------------------------------------------------------
        % Get postscript for straight arrows (used for extra parents)
        %------------------------------------------------------------------

        function [arrowX,arrowY] =getArrowStraight(this,iClass,iParent)
            thisClass = this.classList(iClass);
            parentId = iParent;
                        
            if(parentId>0)
                thisBoxMidX = thisClass.boxX + thisClass.boxW/2;
                thisBoxMidY = thisClass.boxY + thisClass.boxH;
                parentClass = this.classList(parentId);                
                pEndX = parentClass.boxX + parentClass.boxW/2;
                pEndY = parentClass.boxY; 
                
                arrowX = [thisBoxMidX,pEndX];
                arrowY = [thisBoxMidY,pEndY];
            else
                arrowX = [NaN;NaN];
                arrowY = [NaN;NaN];
            end
        end
        
        %------------------------------------------------------------------
        % Get postscript for arrows for compact plotting
        %------------------------------------------------------------------

        function [arrowX,arrowY] =getArrow(this,iClass)
            thisClass = this.classList(iClass);
            parentId = thisClass.parentId;
            level = thisClass.level;
            levelList = this.levelLists{level};
            posInLevel = find(levelList==iClass);
            nInLevel=length(levelList);
            
            % find parents
            if(level>1)
                parentsAbove = unique(this.parentIds(this.levelIds==level));
                nParentsAbove = length(parentsAbove);
                parentPos = find(parentsAbove == parentId);
            else
                nParentsAbove = 0;
                parentPos = 0;
            end
            
            if(parentId>0)
                % find family
                parentClass = this.classList(parentId);
                siblingIds = find(this.parentIds==parentId);
                nSiblings = length(siblingIds);
                siblingPos = find(siblingIds==iClass);
             
                thisBoxMidX = thisClass.boxX + thisClass.boxW/2;
                thisBoxMidY = thisClass.boxY + thisClass.boxH;
                pEndY = parentClass.boxY;
                pBoxMidX = parentClass.boxX + parentClass.boxW/2;
                
                % split horizontal lines for siblings
                if(this.splitLevels)
                    Xshift = parentClass.boxW*siblingPos/(nSiblings+1);
                    if(pBoxMidX > thisBoxMidX)
                        Yshift = this.Ygap*(nInLevel + nParentsAbove - posInLevel - parentPos + 1)/(nInLevel + nParentsAbove - 1) ;
                    else
                        Yshift = this.Ygap*(posInLevel + parentPos - 1)/(nInLevel + nParentsAbove - 1) ;
                    end
                else % don't split lines
                    Xshift = parentClass.boxW/2;
                    if(pBoxMidX > thisBoxMidX)
                        Yshift = this.Ygap*(nParentsAbove - parentPos + 1 )/(nParentsAbove+1);
                    else
                        Yshift = this.Ygap*(parentPos)/(nParentsAbove+1);
                    end
                end
                pEndX = parentClass.boxX + Xshift;
                midY = thisBoxMidY +  Yshift;
                
                arrowX = [thisBoxMidX,thisBoxMidX,pEndX,pEndX];
                arrowY = [thisBoxMidY,midY,midY,pEndY];
            else % no parent means no arrow
                arrowX = [NaN;NaN];
                arrowY = [NaN;NaN];
            end
              
        end
       
        %------------------------------------------------------------------
        % Auxilliary functions for fancy plotting
        % Based on `A node positioning algorithm for general trees'
        % John Q Walker II
        % www.cs.unc.edu/techreports/89-034.pdf
        %------------------------------------------------------------------
               
        function da = apportion(this,v,da)
            lSibId = this.lSibIds(v);
            if(lSibId~=0)
                vIP = v;
                vOP = v;
                vIM = lSibId;
                vOM = getLeftmostSib(this,vIP);
                
                sIP = this.Xmod(vIP);
                sIM = this.Xmod(vIM);
                sOP = this.Xmod(vOP);
                sOM = this.Xmod(vOM);
                
                nextRightVIM = getNextRight(this,vIM);
                nextLeftVIP = getNextLeft(this,vIP);
                
                while(nextRightVIM ~= 0 && nextLeftVIP ~= 0)
                    vIM = nextRightVIM;
                    vIP = nextLeftVIP;
                    vOM = getNextLeft(this,vOM);
                    vOP = getNextRight(this,vOP);
                    
                    this.ancestors(vOP) = v;
                    
                    shift = this.Xprelim(vIM) + sIM ...
                            - (this.Xprelim(vIP) + sIP) ...
                            + getDistance(this,vIM,vIP);
                        
                    if(shift > 0)
                        aTemp = getAncestor(this,vIM,v,da);
                        moveSubtree(this,aTemp,v,shift);
                        sIP = sIP + shift;
                        sOP = sOP + shift;
                    end
                    
                    sIM = sIM + this.Xmod(vIM);
                    sIP = sIP + this.Xmod(vIP);
                    sOM = sOM + this.Xmod(vOM);
                    sOP = sOP + this.Xmod(vOP);
                    
                    nextRightVIM = getNextRight(this,vIM);
                    nextLeftVIP = getNextLeft(this,vIP);
                
                end  % end while
                
                if(nextRightVIM ~= 0 && getNextRight(this,vOP) == 0)
                    this.threads(vOP) = nextRightVIM;
                    this.Xmod(vOP) = this.Xmod(vOP) + sIM - sOP;
                end
                
                if(nextLeftVIP ~= 0 && getNextLeft(this,vOM) == 0)
                    this.threads(vOM) = nextLeftVIP;
                    this.Xmod(vOM) = this.Xmod(vOM) + sIP - sOM;
                    da = v;
                end
        
            end
        
        end
        
        function d = getDistance(this,v,w)
            d = (this.classList(v).boxW + this.classList(w).boxW)/2 ...
               + this.Xgap;
           
            pars = this.parentIds;
            levs = this.levelIds;

            if(pars(v) ~= pars(w) || (levs(v) == 1 && levs(v) == 1))
                d = d + this.Xgap;
            end
        end
            
        function leftmostId = getLeftmostSib(this,classId)
            parent = this.parentIds(classId);
            
            if(parent>0)
                leftmostId = this.firstKidIds(parent);
            else
                leftmostId = 1;
            end
        end
                 
        function nextLeftId = getNextLeft(this,classId)
            if(this.nKids(classId)>0)
                nextLeftId = this.firstKidIds(classId);
            else
                nextLeftId = this.threads(classId);
            end
        end
        
        function nextRightId = getNextRight(this,classId)
            if(this.nKids(classId)>0)
                nextRightId = this.lastKidIds(classId);
            else
                nextRightId = this.threads(classId);
            end
        end
        
        function a = getAncestor(this,vIM,v,defaultAncestor)
         
            aVIM = this.ancestors(vIM);
            
            pAVIM = this.parentIds(aVIM);
            pV = this.parentIds(v);
            
            if(pAVIM == pV)
                a = aVIM;
            else
                a = defaultAncestor;
            end
        end
        
        function num = getNumber(this,classId)
            parent = this.parentIds(classId);

            if(parent>0)
                sibs = this.kidLists{parent};
            else
                sibs = this.levelLists{1};
            end
                 
            num = find(classId==sibs); 
        end
        
        function moveSubtree(this,wM,wP,shift)
            nSubtrees = getNumber(this,wP) - getNumber(this,wM);
            this.changes(wP) = this.changes(wP) - shift/nSubtrees;
            this.shifts(wP) = this.shifts(wP) + shift;
            this.changes(wM) = this.changes(wM) + shift/nSubtrees;
            this.Xprelim(wP) = this.Xprelim(wP) + shift;
            this.Xmod(wP) = this.Xmod(wP) + shift;
        end
        
        function executeShifts(this,v)
            shift = 0;
            change = 0;
            kids = this.kidLists{v};
            nK = length(kids);
            for iKid = nK:-1:1
                kidId = kids(iKid);
                this.Xprelim(kidId) = this.Xprelim(kidId) + shift;
                this.Xmod(kidId) = this.Xmod(kidId) + shift;
                change = change + this.changes(kidId);
                shift = shift + this.shifts(kidId) + change;
            end
        end
        
        function firstWalk(this,classId)          
                                    
            if(this.nKids(classId) == 0)       % is a leaf
                lSibId = this.lSibIds(classId);  % left sibling
                if(lSibId ~= 0)
                    this.Xprelim(classId) = this.Xprelim(lSibId) ...
                         + getDistance(this,classId,lSibId);
                end
            else
                da = this.firstKidIds(classId);  % leftmost child
                
                kids = this.kidLists{classId};
                nK = length(kids);
                
                for iKid = 1:nK
                    kidId = kids(iKid);
                    firstWalk(this,kidId);
                    da = apportion(this,kidId,da);
                end
                
                executeShifts(this,classId);
                
                midpoint = ( this.Xprelim(this.firstKidIds(classId)) ...
                            + this.Xprelim(this.lastKidIds(classId)) )/2 ;
                        
                lSibId = this.lSibIds(classId);  % left sibling
                if(lSibId ~= 0)
                    this.Xprelim(classId) = this.Xprelim(lSibId) ...
                         + getDistance(this,classId,lSibId);
                    this.Xmod(classId) = this.Xprelim(classId) - midpoint;
                else
                    this.Xprelim(classId) = midpoint;
                end
            end
        end
         
        function secondWalk(this, classId, modSum)
            
            thisClass = this.classList(classId);
            
            if(thisClass.level>0)
                thisClass.X = this.Xprelim(classId) + modSum;
                thisClass.Y = this.cumHeight(thisClass.level);
            else
                thisClass.X = NaN;
                thisClass.Y = NaN;
            end
            
            kids = this.kidLists{classId};
            nK = length(kids);

            for iKid = 1:nK
                kidId = kids(iKid);
                secondWalk(this,kidId, modSum + this.Xmod(classId));
            end
        end
    end  % end methods
     
end