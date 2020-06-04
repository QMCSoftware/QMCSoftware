function makeClassDiagram

    workingDir = cd;
    addpath(workingDir);
    %rmpath(genpath(pwd));
    %addpath([workingDir filesep 'Auxiliary' filesep 'UMLDiagram'])
    
    if(exist('D:\','dir'))
        cd 'D:\Sync\Projects\2DCode\2DChebClass\2DLibrary';    
    elseif(exist('/Users/NoldAndreas/','dir'))
        cd '/Users/NoldAndreas/Documents/ContactLineDynamics/Projects/2DCode/2DChebClass/2DLibrary';        
    elseif(exist('/home/bgoddard/','dir'))
        cd '/home/bgoddard/work/MATLAB/Fluids/NComponent2D/DDFT/2DChebClass/2DLibrary';
    elseif(exist('/Users/Ben/','dir'))
        cd '/Users/Ben/work/MATLAB/Fluids/NComponent2D/DDFT/2DChebClass/2DLibrary';    
    end

	dirList = recursiveDirList(pwd);


    nDir = length(dirList);

    fileList = {};
    classNames = {};
    nFiles = 0;
    completeTree = true;
    md=[];
    nonParentClass = [];

    oldPath = path;

    % read in files from directories
    for iDir = 1:nDir
       addpath(dirList{iDir});

       listingM = dir([dirList{iDir} filesep '*.m']);
       nFilesM = length(listingM);

       listingAT = dir([dirList{iDir} filesep '@*']);
       nFilesAT = length(listingAT);

       for iFile = 1:nFilesM
            classNameM = listingM(iFile).name(1:end-2);
            classNames = [classNames,classNameM];  %#ok
       end

       for iFile = 1:nFilesAT
            classNameAT = listingAT(iFile).name(2:end);
            classNames = [classNames,classNameAT];  %#ok
       end

       nFiles = nFiles + nFilesM + nFilesAT;
    end

    % establish which are valid classes
    for iFile = 1:nFiles
        className = classNames{iFile};
        metaDataCmd = ['md  = ?' className ';'];

        try
            eval(metaDataCmd);

        catch err

            if(strcmp(err.identifier, 'MATLAB:class:InvalidSuperClass'))
                completeTree = false;
                nonParentClass = className;
            else
                fprintf(1,'Unknown error:\n');
                fprintf(1,[err.message '\n']);
            end
        end

        if(~isempty(md) && completeTree)                    
            fileList = [fileList className];      %#ok
        end
    end

    % put into alphabetical order
    [~,sortOrder] = sort(lower(fileList));
    fileList = fileList(sortOrder);

    if(~completeTree)
        fileList = {};
    end

    if(isempty(fileList))

        if(completeTree)
            fprintf(1,'Selected directories contain no class files!\n');
        else
            fprintf(1,['Incomplete tree: ' nonParentClass ' missing superclasses\n']);
        end

    else

        options.fancy=true;
        
        tree = Tree(fileList,options);
        saveFile = tree.saveFile;

        open(saveFile);

    end

    path(oldPath);
    
    cd(workingDir);
    
%     if(exist('D:\','dir'))
%         cd 'D:\Sync\Projects\2DCode\2DChebClass';        
%     elseif(exist('/Users/NoldAndreas/','dir'))
%         cd '/Users/NoldAndreas/Documents/ContactLineDynamics/Projects/2DCode/2DChebClass';
%     elseif(exist('/home/bgoddard/','dir'))
%         cd '/home/bgoddard/work/MATLAB/Fluids/NComponent2D/DDFT/2DChebClass';
%     elseif(exist('/Users/Ben/','dir'))
%         cd '/Users/Ben/work/MATLAB/Fluids/NComponent2D/DDFT/2DChebClass';        
%     end
    
    %AddPaths();

end