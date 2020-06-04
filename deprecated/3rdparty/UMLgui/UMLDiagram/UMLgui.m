function UMLgui
% UMLGUI is a gui for creating UML class diagrams
%
% See also:
%   ClassFile Tree ClassGroup
%
% Ben Goddard 12/12/13


    %----------------------------------------------------------------------
    % Set up figure
    %----------------------------------------------------------------------
    
    screenSize = get(0,'ScreenSize');

    fX = (screenSize(3)-600)/2;
    fY = (screenSize(4)-600)/2;
    
    f = figure('Visible','off','Position',[fX,fY,600,600],'Units','pixels',...
       'MenuBar','none','ToolBar','none','Resize','off', ...
       'NumberTitle','off','Name','UML Maker');
   
    set(f,'defaultUIcontrolFontSize',12);
    set(f,'defaultUIcontrolFontName','Arial');
   
    myguidata.handles = guihandles(f);

    % defaults
    textColour = [0.9, 0.9, 0.9];
    editColour = 'white';
    popColour = 'white';
    panelColour = 'white';
    tickColour = 'white';
    msgColour = 'white';
    
    %----------------------------------------------------------------------
    % Set up directory box and buttons
    %----------------------------------------------------------------------

    emptyString = 'Add directories ...';
    
    myguidata.emptyString = emptyString;
    
    myguidata.dirList = [];
    
    dirBoxX   = 10;
    dirBoxY   = 305;
    dirBoxW   = 580;
    dirBoxH   = 285;
    dirBoxSep = 5;
    
    dirButW = 120;
    dirButH = 25;
    dirButX = dirBoxX + dirBoxW/2;
    dirButY = dirBoxY - dirBoxSep - dirButH;
    dirButSep = 20;
    
    % directory box
    uicontrol('Style','listbox','String',emptyString,'Value',1, ...
                 'Min',1,'Max',64,'BackgroundColor','white', ...
                 'Position',[dirBoxX, dirBoxY, dirBoxW, dirBoxH], ...
                 'Tag','hDirList');

    % add button
    uicontrol('Style', 'pushbutton', 'String', 'Add',...
                'Position', [dirButX - 3/2*dirButSep - 2*dirButW, dirButY, dirButW, dirButH], ...
                'Callback', @addDirs, ...
                'Tag','hAddDir');                     
            
    % add with subfolders button
    uicontrol('Style', 'pushbutton', 'String', 'Add + subdirs',...
                'Position', [dirButX - dirButSep/2 - dirButW, dirButY, dirButW, dirButH], ...
                'Callback', @addDirsR, ...
                'Tag','hAddDirR');                     
     
    % delete button
    uicontrol('Style', 'pushbutton', 'String', 'Delete',...
                'Position', [dirButX + dirButSep/2, dirButY, dirButW, dirButH], ...
                'Callback', @delDirs, ...
                'Tag','hDelDir');

    % clear button
    uicontrol('Style', 'pushbutton', 'String', 'Clear all',...
                'Position', [dirButX + 3/2*dirButSep + dirButW, dirButY, dirButW, dirButH], ...
                'Callback', @clearDirs, ...
                'Tag','hClearDir');

     
    %----------------------------------------------------------------------
    % Set up title edit box
    %----------------------------------------------------------------------                    
   
    titleW = dirBoxW;
    titleH = 25;
    titleX = dirBoxX;
    titleY = dirButY - titleH - 5;

    myguidata.title = [];
    myguidata.titleDefault = 'Edit to choose title (currently emtpy)';

    uicontrol('Style','edit','String',myguidata.titleDefault, ...,
                'Max',1,'Min',1,...
                'Position', [titleX,titleY,titleW,titleH], ...
                'BackgroundColor',editColour, ...
                'Enable','inactive', ...
                'ButtonDownFcn', @clearBox, ...
                'Callback',@changeTitle, ...
                'Tag','hTitle');
    
    %----------------------------------------------------------------------
    % Set up message box
    %----------------------------------------------------------------------
    
    msgW = dirBoxW;
    msgH = 18;
    msgX = dirBoxX;
    msgY = titleY - msgH;

    myguidata.msgString = [];

    uicontrol('Style','text','String',myguidata.msgString, ...,
                'Max',1,'Min',1,...
                'Position', [msgX,msgY,msgW,msgH], ...
                'BackgroundColor',msgColour, ...
                'Visible','off', ...
                'Callback',@noMsgEdit, ...
                'Tag','hMsg');
                
    %----------------------------------------------------------------------
    % Set up output file box and browser button
    %----------------------------------------------------------------------
                
    saveW = dirBoxW - 120;
    saveH = 25;
    saveX = dirBoxX;
    saveY = msgY - saveH;

    myguidata.opts.saveFile = [];
    myguidata.saveDefault = 'Edit to choose output file (leave for default), or click here --->';
    
    % output file box
    uicontrol('Style','edit','String', myguidata.saveDefault, ...
                'Max',1,'Min',1,...
                'Position', [saveX,saveY,saveW,saveH], ...
                'BackgroundColor',editColour, ...
                'Enable','inactive', ...
                'ButtonDownFcn', @clearBox, ...
                'Callback',@changeSaveText, ...
                'Tag','hSaveText');

	saveButX = saveX + saveW;
    saveButY = saveY;
    saveButW = 120;
    saveButH = 25;
    
    % browser button
    uicontrol('Style', 'pushbutton', 'String', 'Open browser',...
                'Position', [saveButX, saveButY, saveButW, saveButH], ...
                'Callback',@getSaveFile, ...
                'Tag','hSaveBut');
                 
    %----------------------------------------------------------------------
    % Set up overwrite tickbox
    %----------------------------------------------------------------------
    
    overTextW = 120;
    overTextH = 18;
    overTextX = 30;
    overTextY = 180;

    overTickW = 20;
    overTickH = 20;
    overTickX = overTextX + overTextW;
    overTickY = overTextY;
    
    myguidata.overwriteDefault = 'Overwrite file?';
    
    % overwrite text
    uicontrol('Style','text','String',myguidata.overwriteDefault, ...
                'BackgroundColor',textColour, ...
                'Position',[overTextX, overTextY, overTextW, overTextH], ...
                'Visible','off',...
                'Tag','hOverwriteText');
                 
    % overwrite tickbox                 
    uicontrol('Style','checkbox','Max',1,'Min',0,'Value',0, ...
                'BackgroundColor',tickColour, ...
                'Position',[overTickX, overTickY, overTickW, overTickH], ...
                'Visible','off',...
                'Callback', @setOverwrite, ...
                'Tag','hOverwriteTick');
            
    % default to being able to overwrite as there's no file selected
    myguidata.overwrite = 1;
                     
    %----------------------------------------------------------------------
    % Set up GO button (to make UML)
    %----------------------------------------------------------------------

    goButW = 200;
    goButH = 25;
    goButX = 200;
    goButY = 175;

    uicontrol('Style', 'pushbutton', 'String', 'Create UML diagram',...
                'Position', [goButX, goButY, goButW, goButH], ...
                'Callback', @doUML, ...
                'Enable', 'off', ...
                'Tag','hGo');
                
    %----------------------------------------------------------------------
    % Set up button to open pdf file
    %----------------------------------------------------------------------
                
    openButW = 100;
    openButH = 25;
    openButX = goButX + 250;
    openButY = goButY;

    uicontrol('Style', 'pushbutton', 'String', 'Open PDF file',...
                'Position', [openButX, openButY, openButW, openButH], ...
                'Callback', @openPDF, ...
                'Enable', 'off','Visible','off', ...
                'Tag','hOpenPDF'); 
    
    %----------------------------------------------------------------------
    % Set up font selection
    %----------------------------------------------------------------------
    
    fontPanelX = 12;
    fontPanelY = 10;
    fontPanelW = 220;
    fontPanelH = 145;
    
    fontTextX   = 10;
    fontTextY   = 18;
    fontTextW   = 90;
    fontTextH   = 20;
    fontTextGap = 25;

    fontTSSep = 10;
    
    fontSelX   = fontTextX + fontTextW + fontTSSep;
    fontSelY   = fontTextY+7;
    fontSelW   = 100;
    fontSelH   = 15;
    fontSelGap = fontTextGap;
    
    % font options
    myguidata.fontList = {'Times-Roman','Times-Bold','Times-Italic'};
    fontListDisp = {'Roman','Bold','Italic'};
    
    % Set up font selection panel
    hFontsPanel = uipanel('Parent',f,'Title','Font Selection','FontSize',12,...
                            'BackgroundColor',panelColour,'Units','pixels',...
                            'Position',[fontPanelX, fontPanelY, fontPanelW, fontPanelH], ...
                            'Tag','hFontsPanel');

    %----------------------------------------------------------------------
    % Set up font text for class name
    uicontrol('Parent',hFontsPanel,'Style','text','String','Class Name', ...
                'BackgroundColor',textColour, ...
                'Position',[fontTextX, fontTextY + 3*fontTextGap, fontTextW, fontTextH], ...
                'Tag','hNameFontText');
    
    % Set up font selection for class name            
    uicontrol('Parent',hFontsPanel,'Style','popupmenu', ...
                'BackgroundColor',popColour, ...
                'String',fontListDisp, ...
                'Value',2, ...
                'Position',[fontSelX, fontSelY + 3*fontSelGap, fontSelW, fontSelH], ...
                'Callback', @setNameFont, ...
                'Tag','hNameFont');
    
    % default is bold
    myguidata.opts.nameFont=myguidata.fontList{2};
    
    %----------------------------------------------------------------------
    % Set up font text for properties
    uicontrol('Parent',hFontsPanel,'Style','text','String','Properties', ...
                'BackgroundColor',textColour, ...
                'Position',[fontTextX, fontTextY + 2*fontTextGap, fontTextW, fontTextH], ...
                'Tag','hPropsFontText');
            
    % Set up font selection for properties            
    uicontrol('Parent',hFontsPanel,'Style','popupmenu', ...
                'BackgroundColor',popColour, ...
                'String',fontListDisp, ...
                'Value',1, ...
                'Position',[fontSelX, fontSelY + 2*fontSelGap, fontSelW, fontSelH], ...
                'Callback', @setPropsFont, ...
                'Tag','hPropsFont');
             
    % default is roman
    myguidata.opts.propsFont=myguidata.fontList{1};
    
    %----------------------------------------------------------------------
    % Set up font text for methods         
    uicontrol('Parent',hFontsPanel,'Style','text','String','Methods', ...
                'BackgroundColor',textColour, ...
                'Position',[fontTextX, fontTextY + 1*fontTextGap, fontTextW, fontTextH], ...
                'Tag','hMethodsFontText');

    % Set up font selection for methods
    uicontrol('Parent',hFontsPanel,'Style','popupmenu', ...
                'BackgroundColor',popColour, ...
                'String',fontListDisp, ...
                'Value',1, ...
                'Position',[fontSelX, fontSelY + 1*fontSelGap, fontSelW, fontSelH], ...
                'Callback', @setMethodsFont, ...
                'Tag','hMethodsFont');
    
    % default is roman
    myguidata.opts.methodsFont=myguidata.fontList{1};
    
    %----------------------------------------------------------------------
    % Set up text for font size
    uicontrol('Parent',hFontsPanel,'Style','text','String','Font Size', ...
                'BackgroundColor',textColour, ...
                'Position',[fontTextX, fontTextY + 0*fontTextGap, fontTextW, fontTextH], ...
                'Tag','hFontSizeText');          

    % Set up font size selection
    uicontrol('Parent',hFontsPanel,'Style','popupmenu', ...
                'BackgroundColor',popColour, ...
                'String',{'6','8','10','12','14','16','18','20'}, ...
                'Value',4, ...
                'Position',[fontSelX, fontSelY + 0*fontSelGap, fontSelW, fontSelH], ...
                'Callback', @setFontSize, ...
                'Tag','hFontSize');

    % default is 12
    myguidata.opts.fontSize=12;
    
    %----------------------------------------------------------------------
    % Set up view selection
    %----------------------------------------------------------------------
    
    viewPanelX = fontPanelX + fontPanelW + 10;
    viewPanelY = fontPanelY;
    viewPanelW = 345;
    viewPanelH = fontPanelH;
    
    % view panel
    hViewPanel = uipanel('Parent',f,'Title','View Selection','FontSize',12,...
                            'BackgroundColor','white','Units','pixels',...
                            'Position',[viewPanelX, viewPanelY, viewPanelW, viewPanelH], ...
                            'Tag','hViewPanel');

    gapTextX   = 10;
    gapTextY   = 13;
    gapTextW   = 180;
    gapTextH   = 20;
    gapTextGap = 20;

    gapSelX   = gapTextX + gapTextW + 5;
    gapSelY   = gapTextY + 3;
    gapSelW   = 100;
    gapSelH   = 15;
    gapSelGap = gapTextGap;
    
    gapEditX   = gapSelX + gapSelW + 5;
    gapEditY   = gapTextY ;
    gapEditW   = 30;
    gapEditH   = 20;
    gapEditGap = gapTextGap;
    
    Xgap0 = 50;
    Ygap0 = 50;
    
    
    doTextX   = 10;
    doTextY   = gapTextY + 2*gapSelGap + 10;
    doTextW   = 110;
    doTextH   = 20;
    doTextGap = 20;

    doTSSep = 5;
    
    doSelX   = doTextX + doTextW + doTSSep;
    doSelY   = doTextY + 1;
    doSelW   = 20;
    doSelH   = 20;
    doSelGap = doTextGap;
    
    
    fancyTextX = doSelX + doSelW + 45;
    fancyTextY = doTextY + 1.7*gapTextGap;
    fancyTextH = 20;
    fancyTextW = 100;
    
    fancySelX = doSelX + doSelW + 30;
    fancySelH = 20;
    fancySelY = fancyTextY - fancySelH - 5;
    fancySelW = 130;

    myguidata.fancyList = {'Aesthetic','Compact'};
    myguidata.fancyVals = [1;0];
    myguidata.fancy = 1;
    
    
    %----------------------------------------------------------------------    
    % Set up properties text
    uicontrol('Parent',hViewPanel,'Style','text','String','Properties', ...
                'BackgroundColor',textColour, ...
                'Position',[doTextX, doTextY + 2*doTextGap, doTextW, doTextH], ...
                'Tag','hDoPropsText');
    
    % Set up properties selection
    uicontrol('Parent',hViewPanel,'Style','checkbox','Max',1,'Min',0,'Value',1, ...
                'BackgroundColor',tickColour, ...
                'Position',[doSelX, doSelY + 2*doSelGap, doSelW, doSelH], ...
                'Callback', @setDoProps, ...
                'Tag','hDoProps');
            
    % default is on
    myguidata.opts.doProps = 1;

    %----------------------------------------------------------------------    
    % Set up methods text
    uicontrol('Parent',hViewPanel,'Style','text','String','Methods', ...
                'BackgroundColor',textColour, ...
                'Position',[doTextX, doTextY + 1*doTextGap, doTextW, doTextH], ...
                'Tag','hDoMethodsText');
                     
    % Set up methods selection
    uicontrol('Parent',hViewPanel,'Style','checkbox','Max',1,'Min',0,'Value',1, ...
                'BackgroundColor',tickColour, ...
                'Position',[doSelX, doSelY + 1*doSelGap, doSelW, doSelH], ...
                'Callback', @setDoMethods, ...
                'Tag','hDoMethods');
            
    % default is on
    myguidata.opts.doMethods = 1;
    
    %----------------------------------------------------------------------    
    % Set up inputs text
    uicontrol('Parent',hViewPanel,'Style','text','String','Method Inputs', ...
                'BackgroundColor',textColour, ...
                'Position',[doTextX, doTextY + 0*doTextGap, doTextW, doTextH], ...
                'Tag','hDoInputsText');

    % Set up inputs selection
    uicontrol('Parent',hViewPanel,'Style','checkbox','Max',1,'Min',0,'Value',1, ...
                'BackgroundColor',tickColour, ...
                'Position',[doSelX, doSelY + 0*doSelGap, doSelW, doSelH], ...
                'Callback', @setDoInputs, ...
                'Tag','hDoInputs');
            
    % default is on
    myguidata.opts.doMethodsInputs = 1;

    %----------------------------------------------------------------------        
    % Set up fancy/compact text
    uicontrol('Parent',hViewPanel,'Style','text','String','Tree Layout:', ...
                'BackgroundColor',textColour, ...
                'Position', [fancyTextX, fancyTextY, fancyTextW, fancyTextH], ...
                'Tag','hFancyText');    
    
    % Set up fancy/compact selection            
    uicontrol('Parent',hViewPanel,'Style','popupmenu', ...
                'BackgroundColor',popColour, ...
                'String',myguidata.fancyList, ...
                'Value',1, ...
                'Position',[fancySelX, fancySelY, fancySelW, fancySelH], ...
                'Callback', @setFancy, ...
                'Tag','hFancy');
            
    %----------------------------------------------------------------------    
    % Set up horizontal spacing text
    uicontrol('Parent',hViewPanel,'Style','text','String','Horizontal Spacing (pts)', ...
                'BackgroundColor',textColour, ...
                'Position', [gapTextX, gapTextY + 1*gapTextGap, gapTextW, gapTextH], ...
                'Tag','hXGapSize');
    
    % Set up horizontal spacing slider
    hXGapSlider = uicontrol('Parent',hViewPanel,'Style','slider','Max',500,'Min',0,'Value',Xgap0, ...
                                'Position',[gapSelX, gapSelY  + 1*gapSelGap, gapSelW, gapSelH], ...
                                'Callback', @setXGapSlider, ...
                                'Tag','hXGapSlider');
    addlistener(hXGapSlider,'Value','PostSet',@setXGapSlider);   
    myguidata.opts.Xgap = Xgap0;

    % Set up horizontal spacing edit box
    uicontrol('Parent',hViewPanel,'Style','edit','String',num2str(Xgap0), ...
                'BackgroundColor',textColour, ...
                'Position', [gapEditX, gapEditY + 1*gapEditGap, gapEditW, gapEditH], ...
                'Callback', @setXGapEdit, ...
                'Tag','hXGapEdit');
    
    %----------------------------------------------------------------------    
    % Set up vertical spacing selection text        
    uicontrol('Parent',hViewPanel,'Style','text','String','Vertical Spacing (pts)', ...
                'BackgroundColor',textColour, ...
                'Position', [gapTextX, gapTextY + 0*gapTextGap, gapTextW, gapTextH], ...
                'Tag','hYGapText');
    
    % Set up vertical spacing slider
    hYGapSlider = uicontrol('Parent',hViewPanel,'Style','slider','Max',500,'Min',0,'Value',Ygap0, ...
                                'Position',[gapSelX, gapSelY + 0*gapSelGap, gapSelW, gapSelH], ...
                                'Callback', @setYGapSlider, ...
                                'Tag','hYGapSlider');
    addlistener(hYGapSlider,'Value','PostSet',@setYGapSlider);
    myguidata.opts.Ygap = Ygap0;
    
    % Set up vertical spacing edit box              
    uicontrol('Parent',hViewPanel,'Style','edit','String',num2str(Ygap0), ...
                'BackgroundColor',textColour, ...
                'Position', [gapEditX, gapEditY + 0*gapEditGap, gapEditW, gapEditH], ...
                'Callback', @setYGapEdit, ...
                'Tag','hYGapEdit');
        
    
    %----------------------------------------------------------------------
    % Make the GUI visible.
    %----------------------------------------------------------------------
    
    myguidata.myguihandles = guihandles(f);
    guidata(f,myguidata);
    set(f,'Visible','on')
    
   
    %----------------------------------------------------------------------
    % Functions start here
    %----------------------------------------------------------------------


    %----------------------------------------------------------------------
    % Clear an edit box
    %----------------------------------------------------------------------

    function clearBox(hBox,~)

        status = get(hBox,'Enable');
        
        if(strcmp(status,'inactive'))
            set(hBox, 'String', [], 'Enable', 'on');
            uicontrol(hBox); 
        end

    end

    %----------------------------------------------------------------------
    % Change the save file text
    %----------------------------------------------------------------------
    
    function changeSaveText(~,~)
        tempguidata = guidata(f);
        
        saveText = tempguidata.myguihandles.hSaveText;
        newSave = get(saveText,'String');

        % if the new save file is empty the default
        if(isempty(newSave) || strcmp(newSave,tempguidata.saveDefault) )
            % set to the default
            set(saveText,'String',tempguidata.saveDefault);
            set(saveText,'Enable','inactive');
            
            % reset message box to remove file-specific announcements
            resetMsg();
            
            % make overwrite invisible
            toggleOverwrite(false);
            
            % if there are files in the dirList, make the GO button active
            if(~isempty(tempguidata.dirList))
                hUML  = tempguidata.myguihandles.hGo;
                set(hUML,'Enable','on');
                tempguidata.overwrite = 1;
            end
            
            % pass default file to the Tree
            tempguidata.opts.saveFile = [];
        else
            % use this new file
            tempguidata.opts.saveFile = newSave;
            
            % check existence of file
            checkSaveFile(tempguidata.opts.saveFile);
        end
        
        % disable pdf button as we no longer have a corresponding pdf file
        togglePdf(false);
        
        guidata(f,tempguidata);
    end

    %----------------------------------------------------------------------
    % Get save file from browser
    %----------------------------------------------------------------------

    function getSaveFile(~,~)
       
        sFile = uigetfileordir;
        if(~isempty(sFile))
            % update save file text
            tempguidata = guidata(f);
            saveText = tempguidata.myguihandles.hSaveText;
            set(saveText,'String',sFile,'Enable','on');

            tempguidata.opts.saveFile = sFile;

            guidata(f,tempguidata);
            
            % turn off pdf button
            togglePdf(false);
            
            % check existence of new file
            checkSaveFile(tempguidata.opts.saveFile);
        end
            
    end

    %----------------------------------------------------------------------
    % Check that the file selected is a pdf file and that the output
    % directory exists
    %----------------------------------------------------------------------

    function checkSaveFile(saveFile)

        tempguidata = guidata(f);
        msgBox = tempguidata.myguihandles.hMsg;
        existence = exist(saveFile); %#ok
        
        if(existence == 2)  % existing file
            [~,~,ext] = fileparts(saveFile);
            % if not a pdf file
            if(~strcmp(ext,'.pdf'))
                set(msgBox,'String','Please choose valid .pdf file','Visible','on');
                toggleOverwrite(false);
                tempguidata.overwrite = 0;
                guidata(tempguidata);
            else
                % set overwrite of selected file
                toggleOverwrite(true);
            end
        elseif(existence == 7)  % existing directory so use default file here
            toggleOverwrite(false);
            resetMsg();
        else
            toggleOverwrite(false);
            [~,~,ext] = fileparts(saveFile);
            if(isempty(ext))  % if no file, i.e. a directory
                set(msgBox,'String','Make new directory?','Visible','on','BackgroundColor','yellow');
                toggleOverwrite(true);
                oText = tempguidata.myguihandles.hOverwriteText;
                set(oText,'String','Create directory');
            elseif(~strcmp(ext,'.pdf')) % not a pdf file
                set(msgBox,'String','Please choose valid .pdf file','Visible','on', ...
                    'BackgroundColor','yellow');
            end
        end
        
    end
    
    %----------------------------------------------------------------------
    % Prevent editing of the message text
    %----------------------------------------------------------------------

    function noMsgEdit(~,~)
        tempguidata = guidata(f);
        hMes = tempguidata.myguihandles.hMsg;
        mes = tempguidata.msgString;
        set(hMes,'String',mes);
    end

    %----------------------------------------------------------------------
    % Reset the message text
    %----------------------------------------------------------------------

    function resetMsg()
        tempguidata = guidata(f);
        msgBox = tempguidata.myguihandles.hMsg;
        set(msgBox,'String','','Visible','off','BackgroundColor','white');
    end

    %----------------------------------------------------------------------
    % Switch visibility of overwrite selection
    %----------------------------------------------------------------------

    function toggleOverwrite(vis)
        tempguidata = guidata(f);
        
        oText = tempguidata.myguihandles.hOverwriteText;
        oTick = tempguidata.myguihandles.hOverwriteTick;
        
        if(vis)
            set(oText,'Visible','on','BackgroundColor','white', ...
                'String',tempguidata.overwriteDefault);
            set(oTick,'Visible','on','Value',0);
            tempguidata.overwrite = 0;
        else
            set(oText,'Visible','off','BackgroundColor','white', ...
                'String',tempguidata.overwriteDefault);
            set(oTick,'Visible','off','Value',0);
            tempguidata.overwrite = 1;
        end
        
        guidata(f,tempguidata);
    end

    %----------------------------------------------------------------------
    % Switch visibility of overwrite selection
    %----------------------------------------------------------------------

    function setOverwrite(~,~)
        
        tempguidata = guidata(f);
        hTick = tempguidata.myguihandles.hOverwriteTick;
        val = get(hTick,'Value');
        
        if(val)
            resetMsg();
            hText = tempguidata.myguihandles.hOverwriteText;
            status = get(hText,'BackgroundColor');
            % if the status bar is yellow, i.e. we're waiting to verify the
            % overwrite
            if(status(1) == 1 && status(2) == 1 && status(3) == 0);
                set(hText,'BackgroundColor','white');
                UMLbut = tempguidata.myguihandles.hGo;
                set(UMLbut,'Enable','on');
            end
        end
        
        tempguidata.overwrite = val;
        guidata(f,tempguidata);
        
    end

    %----------------------------------------------------------------------
    % Change title text
    %----------------------------------------------------------------------

    function changeTitle(~,~)
        tempguidata = guidata(f);
        
        titleText = tempguidata.myguihandles.hTitle;
        newTitle = get(titleText,'String');
        if(isempty(newTitle))
            % if empty switch to the default
            set(titleText,'String',tempguidata.titleDefault);
            set(titleText,'Enable','inactive');
            tempguidata.opts.title = [];
        else
            tempguidata.opts.title = newTitle;
        end
        
        togglePdf(false);
        
        guidata(f,tempguidata);
    end

    %----------------------------------------------------------------------
    % Open the created pdf file
    %----------------------------------------------------------------------
    
    function openPDF(~,~)
        tempguidata = guidata(f);
        open([tempguidata.opts.saveFile]);
    end
    
    %----------------------------------------------------------------------
    % Toggle visibility of the open pdf button
    %----------------------------------------------------------------------

    function togglePdf(vis)
        tempguidata = guidata(f);
        openBut = tempguidata.myguihandles.hOpenPDF;
        if(vis)
            set(openBut,'Enable','on','Visible','on');
        else
            set(openBut,'Enable','off','Visible','off');
        end 
    end

    %----------------------------------------------------------------------
    % Add directories without recursion
    %----------------------------------------------------------------------

    function addDirs(~,~)
        addDirsGeneral(false)
    end

    %----------------------------------------------------------------------
    % Add directories with recursion
    %----------------------------------------------------------------------
    
    function addDirsR(~,~)
        addDirsGeneral(true)
    end

    %----------------------------------------------------------------------
    % Add list of directories
    %----------------------------------------------------------------------

    function addDirsGeneral(rDir)
        % rDir determines whether it recurses through subdirectories
        
        % get directory list
        newDirList = uigetdirs;
        
        tempguidata = guidata(f);
        hList = tempguidata.myguihandles.hDirList;
        hUML  = tempguidata.myguihandles.hGo;
        emptyS = tempguidata.emptyString;
        
        oldDirList = get(hList,'String');
        
        if(~isempty(newDirList))
                            
            if(rDir) % do recursion
                nDir = length(newDirList);
                subDirList = {};
                for iDir = 1:nDir
                    thisSubDirList = recursiveDirList(newDirList{iDir});
                    subDirList = [subDirList; thisSubDirList];    %#ok
                end
                newDirList = subDirList';
            end
            
            % if the list was empty then add the new directories
            if(strcmp(oldDirList,emptyS))  
                set(hList,'String',newDirList')
                tempguidata.dirList = newDirList';
                set(hUML,'Enable','on');
            else
                % if the list wasn't empty then add only newly selected
                % directories, removing duplicates
                tempDirList = [oldDirList; newDirList'];
                uniqueDirList = unique(tempDirList);
                set(hList,'String',uniqueDirList)                    
                tempguidata.dirList = uniqueDirList;
                set(hUML,'Enable','on');
            end
            
            msgBox = tempguidata.myguihandles.hMsg;
            set(msgBox,'Visible','off');
            
            togglePdf(false);

            guidata(f,tempguidata);

        end
        
         
    end

    %----------------------------------------------------------------------
    % Delete selected directories
    %----------------------------------------------------------------------
    
    function delDirs(~,~)
        
        tempguidata = guidata(f);
        hList = tempguidata.myguihandles.hDirList;
        hUML  = tempguidata.myguihandles.hGo;
        emptyS = tempguidata.emptyString;
        
        toDel = get(hList,'Value');
        tempDirList = get(hList,'String');
        
        if(~strcmp(tempDirList,emptyS))
            % if the list isn't empty, set the ones to be deleted to []
            tempDirList(toDel) = [];

            % if it's now empty, set it to the default
            if(isempty(tempDirList))
                tempDirList = emptyS;
                tempguidata.dirList = [];
                set(hUML,'Enable','off');
            end
            
            % select first directory in the list and update
            set(hList,'Value',1);
            set(hList,'String',tempDirList);
            tempguidata.dirList = tempDirList;
        end
        
        msgBox = tempguidata.myguihandles.hMsg;
        set(msgBox,'Visible','off');  
        
        guidata(f,tempguidata);
        
        togglePdf(false); 
        
    end


    %----------------------------------------------------------------------
    % Clear whole list of directories
    %----------------------------------------------------------------------

    function clearDirs(~,~)
        tempguidata = guidata(f);
        hList = tempguidata.myguihandles.hDirList;
        hUML  = tempguidata.myguihandles.hGo;
        emptyS = tempguidata.emptyString;
        
        set(hList,'Value',1);
        set(hList,'String',emptyS);
        tempguidata.dirList = [];
        set(hUML,'Enable','off');
        
        msgBox = tempguidata.myguihandles.hMsg;
        set(msgBox,'Visible','off');  
        
        guidata(f,tempguidata);
        
        togglePdf(false); 

    end


    %----------------------------------------------------------------------
    % Set value of X gap slider
    %----------------------------------------------------------------------
    
    function setXGapSlider(~,~)
        
        tempguidata = guidata(f);
        hSlider = tempguidata.myguihandles.hXGapSlider;
        hEdit = tempguidata.myguihandles.hXGapEdit;
        
        val = get(hSlider,'Value');
        val = round(val);
        
        % update edit box
        set(hEdit,'String',num2str(val));
        
        tempguidata.opts.Xgap=val;
        guidata(f,tempguidata);
        
        togglePdf(false);
    end

    %----------------------------------------------------------------------
    % Set value of X gap edit box
    %----------------------------------------------------------------------
    
    function setXGapEdit(~,~)
        
        tempguidata = guidata(f);
        hSlider = tempguidata.myguihandles.hXGapSlider;
        hEdit = tempguidata.myguihandles.hXGapEdit;
        
        valStr = get(hEdit,'String');
        val = str2double(valStr);
        
        % check within range
        minVal = get(hSlider,'Min');
        maxVal = get(hSlider,'Max');
        
        if(isempty(val))
            val = get(hSlider,'Value');
        end
        if(val<minVal)
            val = minVal;
        end
        if(val>maxVal)
            val = maxVal;
        end
        
        val = round(val);
        
        % update edit box to rounded value
        set(hEdit,'String',num2str(val));
        
        % update slider
        set(hSlider,'Value',val);
        
        tempguidata.opts.Xgap=val;
        guidata(f,tempguidata);
        
        togglePdf(false);
    end
    
    %----------------------------------------------------------------------
    % Set value of Y gap slider
    %----------------------------------------------------------------------
    
    function setYGapSlider(~,~)
        tempguidata = guidata(f);
        hSlider = tempguidata.myguihandles.hYGapSlider;
        hEdit = tempguidata.myguihandles.hYGapEdit;
        
        val = get(hSlider,'Value');
        val = round(val);
        
        %update edit box
        set(hEdit,'String',num2str(val));
        
        tempguidata.opts.Ygap=val;
        guidata(f,tempguidata);
        
        togglePdf(false);
    end
    
    %----------------------------------------------------------------------
    % Set value of Y gap edit box
    %----------------------------------------------------------------------
   
    function setYGapEdit(~,~)
        
        tempguidata = guidata(f);
        hSlider = tempguidata.myguihandles.hYGapSlider;
        hEdit = tempguidata.myguihandles.hYGapEdit;
        
        valStr = get(hEdit,'String');
        val = str2double(valStr);
        
        % check within range
        minVal = get(hSlider,'Min');
        maxVal = get(hSlider,'Max');
        
        if(isempty(val))
            val = get(hSlider,'Value');
        end
        if(val<minVal)
            val = minVal;
        end
        if(val>maxVal)
            val = maxVal;
        end
        
        val = round(val);
        
        % update edit box to rounded value
        set(hEdit,'String',num2str(val));
        
        % update slider
        set(hSlider,'Value',val);
        
        tempguidata.opts.Ygap=val;
        guidata(f,tempguidata);
        
        togglePdf(false);
        
    end

    %----------------------------------------------------------------------
    % Set class name font
    %----------------------------------------------------------------------

    function setNameFont(~,~)
       
        tempguidata = guidata(f);
        hFont = tempguidata.myguihandles.hNameFont;
        fontList = tempguidata.fontList;
        
        font = get(hFont,'Value');
        
        fontText = fontList{font};
        tempguidata.opts.nameFont = fontText;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false);
    end

    %----------------------------------------------------------------------
    % Set properties font
    %----------------------------------------------------------------------

    function setPropsFont(~,~)
       
        tempguidata = guidata(f);
        hFont = tempguidata.myguihandles.hPropsFont;
        fontList = tempguidata.fontList;
        
        font = get(hFont,'Value');
        
        fontText = fontList{font};
        tempguidata.opts.propsFont = fontText;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
     
    end

    %----------------------------------------------------------------------
    % Set methods font
    %----------------------------------------------------------------------

    function setMethodsFont(~,~)
       
        tempguidata = guidata(f);
        hFont = tempguidata.myguihandles.hMethodsFont;
        fontList = tempguidata.fontList;
        
        font = get(hFont,'Value');
        
        fontText = fontList{font};
        tempguidata.opts.methodsFont = fontText;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
     
    end

    %----------------------------------------------------------------------
    % Set font size
    %----------------------------------------------------------------------

    function setFontSize(~,~)
        
        tempguidata = guidata(f);
        hSize = tempguidata.myguihandles.hFontSize;
        
        sizeVal = get(hSize,'Value');
        size = 4 + (sizeVal * 2);  % see default list of values
        
        tempguidata.opts.fontSize = size;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
    end

    %----------------------------------------------------------------------
    % Set properties on or off
    %----------------------------------------------------------------------

    function setDoProps(~,~)
        
        tempguidata = guidata(f);
        hDo = tempguidata.myguihandles.hDoProps;
        
        val = get(hDo,'Value');
        
        tempguidata.opts.doProps = val;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
        
    end

    %----------------------------------------------------------------------
    % Set methods on or off
    %----------------------------------------------------------------------

    function setDoMethods(~,~)
        
        tempguidata = guidata(f);
        hDo = tempguidata.myguihandles.hDoMethods;
        hDoI = tempguidata.myguihandles.hDoInputs;
        
        val = get(hDo,'Value');
        
        % toggle inputs selection
        if(val)
            set(hDoI,'Enable','on','Visible','on');
        else
            set(hDoI,'Enable','off','Visible','off');
        end
         
        tempguidata.opts.doMethods = val;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
        
    end

    %----------------------------------------------------------------------
    % Set inputs on or off
    %----------------------------------------------------------------------

    function setDoInputs(~,~)
        
        tempguidata = guidata(f);
        hDo = tempguidata.myguihandles.hDoInputs;
        
        val = get(hDo,'Value');
        
        tempguidata.opts.doMethodsInputs = val;
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false); 
        
    end

    %----------------------------------------------------------------------
    % Set printing to be fancy or compact
    %----------------------------------------------------------------------

    function setFancy(~,~)
        tempguidata = guidata(f);
        hFancy = tempguidata.myguihandles.hFancy;
        
        fancyVal = get(hFancy,'Value');
        
        tempguidata.opts.fancy = myguidata.fancyVals(fancyVal);
        guidata(f,tempguidata);
        
        % have changed options so need to create a new pdf
        togglePdf(false);
    end



    %----------------------------------------------------------------------
    % Make the UML by calling the appropriate Tree
    %----------------------------------------------------------------------

    function doUML(~,~)
        % check save file as edit box detection is dodgy
        tempguidata = guidata(f);
        saveFile = tempguidata.opts.saveFile;
        if(~tempguidata.overwrite)
            if(~isempty(saveFile))
                checkSaveFile(saveFile);
            else
                toggleOverwrite(false);
            end
        end

        % get handles
        tempguidata = guidata(f);
        addBut = tempguidata.myguihandles.hAddDir;
        addRBut = tempguidata.myguihandles.hAddDirR;
        delBut = tempguidata.myguihandles.hDelDir;
        clearBut = tempguidata.myguihandles.hClearDir;
        UMLBut = tempguidata.myguihandles.hGo;
        msgBox = tempguidata.myguihandles.hMsg;
        saveText = tempguidata.myguihandles.hSaveText;
        oText = tempguidata.myguihandles.hOverwriteText;

        % if overwrite is false then make sure it's validated
        if(~tempguidata.overwrite)
            tempguidata.msgString = 'Please verify overwrite of file';
            set(msgBox,'String',tempguidata.msgString);
            set(msgBox,'BackgroundColor','yellow');
            set(msgBox,'Visible','on');
            set(oText,'BackgroundColor','yellow');
            
        else % overwrite ok
            
            set(addBut,'Enable','off');
            set(addRBut,'Enable','off');
            set(delBut,'Enable','off');
            set(clearBut,'Enable','off');
            set(UMLBut,'Enable','off');
            togglePdf(false);
            set(msgBox,'Visible','off');

            dirList = tempguidata.dirList;

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
                        set(msgBox,'String','Unknown error', ...
                            'BackgroundColor','red','Visible','on');
                    end
                end

                if(~isempty(md) && completeTree)                    
                    fileList = [fileList className];      %#ok
                end
            end
            
            % put into alphabetical order
            [~,sortOrder] = sort(lower(fileList));
            fileList = fileList(sortOrder);
            
            % if we're missing files then default to empty
            if(~completeTree)
                fileList = {};
            end

            % and display an error
            if(isempty(fileList))

                if(completeTree)
                    tempguidata.msgString = 'Selected directories contain no class files!';
                    set(msgBox,'String',tempguidata.msgString,'BackgroundColor','red','Visible','on');
                else
                    set(msgBox,'String',['Incomplete tree: ' nonParentClass ' missing superclasses'], ...
                                'BackgroundColor','red','Visible','on');
                end

                savedUML = false; % didn't complete

            else % we have a valid file list

                % this does all the heavy lifting:
                tree = Tree(fileList,tempguidata.opts);
                
                % update message box to location of file
                saveFile = tree.saveFile;
                tempguidata.msgString = {'Diagram saved to:'};
                set(msgBox,'String',tempguidata.msgString);
                set(msgBox,'BackgroundColor','green');
                set(msgBox,'Visible','on');
                set(UMLBut,'Enable','on');

                set(saveText,'String',saveFile);
                tempguidata.opts.saveFile = saveFile;

                savedUML = true; % did complete

            end

            % restore path
            path(oldPath);

            guidata(f,tempguidata);

            % reset buttons to on
            set(addBut,'Enable','on');
            set(addRBut,'Enable','on');
            set(delBut,'Enable','on');
            set(clearBut,'Enable','on');
            
            % toggle pdf button and overwrite visible
            if(savedUML)
                togglePdf(true);
                toggleOverwrite(true);
                set(saveText,'Enable','on');
            end

        end
    
    end

end