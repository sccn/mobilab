% The role of the class mobilabApplication is to control the graphical
% user  interface (gui). The gui is created on execution time by embedding 
% a Java JTree component in the main window. This interactive tree is created
% to display parent-child relationships between the objects provided by the
% dataSource. Then, a context menu is constructed per object (element on the
% tree) exposing different options for computation, annotation, and visualization.
% 
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Apr-2011

classdef mobilabApplication < handle
    properties
        allStreams                  % Data source object containing a list of stream objects.
        preferences
    end
    properties(SetAccess=protected)
        path = '';                   % Path to the directory where MoBILAB toolbox is being installed.
        doc = ''                     % URL of the documentation online.
    end
    properties(SetAccess=protected,Hidden = true)
        statusBar
        progressBar
    end
    methods
        function obj = mobilabApplication(allStreams)
            if nargin < 1, allStreams = [];end
            p = fileparts(which('runmobilab'));
            if isempty(p)
                p = fileparts(which('eeglab'));
                p = [p filesep 'plugins' filesep 'mobilab'];
            end
            obj.path = p;
            obj.initApplication;
            obj.allStreams = allStreams;
            obj.doc = 'http://sccn.ucsd.edu/wiki/MoBILAB';
            addpath([obj.path filesep 'glibmobi']);
            addpath([obj.path filesep 'gui']);
            addpath([obj.path filesep 'eeglabInterface']);
            dependencyTree = getDirectoryTree([obj.path filesep 'dependency']);
            addDependency2Path(dependencyTree);
        end
        %%
        function set.allStreams(obj,allStreams)
            if ~isa(allStreams,'dataSource') && ~isempty(allStreams)
                warning('MoBILAB:noDataSource','MoBILAB''s property ''allStreams'' has to be a descendant of the class ''dataSource''.')
                return;
            end
            obj.allStreams = allStreams;
        end
        %%
        function batch(obj,functionHandle,folders)
            if nargin < 2, error('Not enough input arguments.');end
            if ~exist(scriptFile,'file'), error('MoBILAB:noScriptFound','Cannot find the script file.');end
            try
                functionHandle = eval(['@' functionHandle]);
                n = length(folders);
                matlabpool open
                parfor it=1:n
                    functionHandle(folders{it});
                end
                matlabpool close
            catch ME
                ME.rethrow;
            end
        end
        %%
        function delete(obj)
            [isActive,figureHandle] = obj.isGuiActive;
            if isActive, delete(figureHandle);end
            %warning off all
            %rmpath(genpath([obj.path filesep 'dependency']));
            %warning on all
            %rmpath(genpath([obj.path filesep 'eeglabInterface']));
            %rmpath(genpath([obj.path filesep 'glibmobi']));
            %rmpath(genpath([obj.path filesep 'gui']));
            configuration = obj.preferences; %#ok
            save(fullfile(getHomeDir,'.mobilab.mat'),'configuration');
            try delete(obj.allStreams);end %#ok 
        end
        %%
        function applicationClose(obj)
            delete(obj);
            evalin('base','clear mobilab');
        end
        %%
        function onlineHelp(obj,url)
            if nargin < 2, url = obj.doc;end
            try
                [~,h] = web(url,'-browser'); %#ok
                return;
            catch ME
                if strcmp(ME.identifier,'MATLAB:unassignedOutputs')
                    status = system(['firefox ' url]);
                    if status, status = system(['google-chrome ' url]);end
                    if status, status = system(['chromium-browser ' url]);end
                    if ~status, return;end
                end     
            end
            web(url);
        end
        %%
        function setPreferences(obj)
           
            stickFigure0 = obj.preferences.mocap.stickFigure;
            stickFigure1 = fullfile(obj.path,'data','sccnOnePersonStickFigure.mat');
            stickFigure2 = fullfile(obj.path,'data','sccnTwoPeopleStickFigure.mat');
            stickFigure = unique({stickFigure0,stickFigure1,stickFigure2});
            
            tmpHeadModels = pickfiles([obj.path filesep 'data'],'headModel');
            headModels = cell(size(tmpHeadModels,1),1);
            rmThis = zeros(size(headModels));
            for it=1:size(tmpHeadModels,1)
                headModels{it} = deblank(tmpHeadModels(it,:));
                if ~isempty(strfind(headModels{it},'.svn')), rmThis(it) = it;end
            end
            rmThis(rmThis==0)=[];
            headModels(rmThis) = [];
            
            oldGui = obj.preferences.gui;
            
            prefObj = [...
                PropertyGridField('username',obj.preferences.username,'Category','User info','DisplayName','Username','Description','Your name. The files resulting from your analysis will be owned by you.'),...
                PropertyGridField('organization',obj.preferences.organization,'Category','User info','DisplayName','Organization','Description','Organization where you belong, for instance, UCSD, NCTU, UTSA, etc..'),...
                PropertyGridField('email',obj.preferences.email,'Category','User info','DisplayName','email','Description','Your email (optional). This allows your collaborators contact you regarding the work you''ve done, this info NEVER will be used by MoBILAB software for other porpuse different than that.'),...
                PropertyGridField('backgroundColor',obj.preferences.gui.backgroundColor,'Category','GUI','DisplayName','BackgroundColor','Description',...
                'Backgroud color for MoBILAB''s guis (set it to [0.76 0.77 1] to get EEGLAB''s color).')...
                PropertyGridField('buttonColor',obj.preferences.gui.buttonColor,'Category','GUI','DisplayName','Botton color','Description','')...
                 PropertyGridField('fontColor',obj.preferences.gui.fontColor,'Category','GUI','DisplayName','FontColor','Description',...
                'Set to false and all graphic outputs (waitbars and guis) will be blocked. This is useful if you''re running scripts then you don''t get windows poping up all the time.')...
                PropertyGridField('interpolation', obj.preferences.mocap.interpolation,'Type',PropertyType('char', 'row', {'pchip','linear','nearest','spline'}),...
                'Category', 'Mocap', 'DisplayName', 'Interpolation method','Description',...
                'Interpolation method to fill-in occluded markers at certain time point (spline methods are recommended: ''pchip'', and ''spline'').')...
                PropertyGridField('smoothing', obj.preferences.mocap.smoothing,'Type',PropertyType('char', 'row', {'sgolay','moving','lowess','loess','rlowess','rloess'}),...
                'Category', 'Mocap', 'DisplayName', 'Smoothing method','Description',...
                'Method to smooth out each channel of mocap data (''sgolay'' is the option good and slow and ''moving'' is the one fast and poor).')...
                PropertyGridField('lowpassCutoff', obj.preferences.mocap.lowpassCutoff, 'Category', 'Mocap', 'DisplayName', 'Lowpass cutoff','Description','')...
                PropertyGridField('derivationOrder', obj.preferences.mocap.derivationOrder, 'Category', 'Mocap', 'DisplayName', 'Derivative order','Description',...
                'Specifies the maximum order of mocap time derivatives, set it to 3 will compute: 1) velocity, 2) acceleration, and 3) jerk.')...
                PropertyGridField('stickFigure', obj.preferences.mocap.stickFigure,'Type',PropertyType('char','row',stickFigure),'Category','Mocap',...
                'DisplayName', 'Mocap stick figure','Description','')...
                PropertyGridField('bodyModel', obj.preferences.mocap.bodyModel,'Category','Mocap','DisplayName', 'Body model','Description','Mat file containing the variables connectivity (connectivity matrix of the body parts/nodes) and nodeLabels (cell array with the names of the body parts).')...
                PropertyGridField('resampleMethod', obj.preferences.eeg.resampleMethod,'Type',PropertyType('char','row', {'linear','nearest','spline','pchip'}),'Category','EEG',...
                'DisplayName','Resampling method','Description','Resampling is done in two steps, 1) lowpass at a half of the new sampling frequency, and 2) interpolation on the new time axis.')...
                PropertyGridField('filterType', obj.preferences.eeg.filterType,'Type',PropertyType('char','row', {'lowpass','highpass','bandpass'}),'Category','EEG',...
                'DisplayName','Bandpass cutoff','Description','')...
                PropertyGridField('cutoff', obj.preferences.eeg.cutoff,'Category','EEG','DisplayName','Filter cutoff frequencies','Description','')...
                PropertyGridField('headModel',obj.preferences.eeg.headModel,'Type',PropertyType('char','row', headModels),'Category','EEG','DisplayName','Head model','Description',...
                'Mat file containing the array of surfaces defining different layers of tissue in certain anatomical space (could be MNI, ICBM, or other).')...
                PropertyGridField('tmpDirectory', obj.preferences.tmpDirectory,'DisplayName','Temp directory','Description',...
                'Directory where save temporary files during MoBILAB''s session.')];
                   
            prefObj = prefObj.GetHierarchy();
            
            % create figure
            hFigure = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none','Units','pixels','userData',0);
            position = get(hFigure,'position');
            set(hFigure,'position',[position(1) position(2)-50 420 570]);
            hPanel = uipanel(hFigure,'Title','','BackgroundColor','white','Units','pixels','Position',[0 77 418 490],'BorderType','none');
            %g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            g = PropertyGrid(hPanel,'Properties', prefObj,'Position', [0 0 1 1]);
            uicontrol(hFigure,'Position',[180 15 80 30],'String','Cancel','ForegroundColor',obj.preferences.gui.fontColor,...
                    'BackgroundColor',obj.preferences.gui.buttonColor,'Callback',@cancelCallback);
            uicontrol(hFigure,'Position',[290 15 80 30],'String','Save','ForegroundColor',obj.preferences.gui.fontColor,...
                    'BackgroundColor',obj.preferences.gui.buttonColor,'Callback',@okCallback);
            uiwait(hFigure); % wait for figure to close
            
            if ~ishandle(hFigure), return;end
            if ~get(hFigure,'userData')
                close(hFigure);
                return;
            end
            close(hFigure);
            drawnow
            
            val = g.GetPropertyValues();
            obj.preferences.username = val.username;
            obj.preferences.organization = val.organization;
            obj.preferences.email = val.email;
            
            obj.preferences.gui.fontColor = val.fontColor;
            obj.preferences.gui.backgroundColor = val.backgroundColor;
            obj.preferences.gui.buttonColor = val.buttonColor;
            
            obj.preferences.mocap.interpolation = val.interpolation;
            obj.preferences.mocap.smoothing = val.smoothing;
            obj.preferences.mocap.lowpassCutoff = val.lowpassCutoff;
            obj.preferences.mocap.derivationOrder = val.derivationOrder;
            obj.preferences.mocap.stickFigure = val.stickFigure;
            obj.preferences.mocap.bodyModel = val.bodyModel;
            
            obj.preferences.eeg.resampleMethod = val.resampleMethod;
            obj.preferences.eeg.filterType = val.filterType;
            obj.preferences.eeg.cutoff = val.cutoff;
            obj.preferences.eeg.headModel = val.headModel;
            obj.preferences.tmpDirectory = val.tmpDirectory;
            
            configuration = obj.preferences; %#ok
            save(fullfile(getHomeDir,'.mobilab.mat'),'configuration');
                
            reDraw = any(obj.preferences.gui.fontColor ~= oldGui.fontColor);
            reDraw = reDraw | any(obj.preferences.gui.backgroundColor ~= oldGui.backgroundColor);
            reDraw = reDraw | any(obj.preferences.gui.buttonColor ~= oldGui.buttonColor);
            [isActive,figureHandle] = obj.isGuiActive();
            if isActive && reDraw, close(figureHandle); obj.gui;end
        end
        %%
        function figureHandle = gui(obj,callback)
            if nargin < 2, callback = 'dispNode_Callback';end
            
            [isActive,figureHandle] = obj.isGuiActive();
            if length(figureHandle) > 1, 
                try close(figureHandle(2:end));end;%#ok
                figureHandle = figureHandle(1);
            end 
            
            if ~isActive
                figureHandle = figure('MenuBar','none','Toolbar','None','NumberTitle','off','Tag','mobilabApplicationGUI','Visible','off',...
                    'Units','Pixels','Color',obj.preferences.gui.backgroundColor);
                position = get(figureHandle,'Position');
                set(figureHandle,'Position',[position(1:2) 418 418]);
                
                % creating the menu
                hFile = uimenu('Label','File','Parent',figureHandle);
                hImport = uimenu(hFile,'Label','Import data');
                uimenu(hImport,'Label','From file','Callback','mobilab.loadDataWizard(''file'');mobilab.gui;');
                uimenu(hImport,'Label','From folder (concat. files)','Callback','mobilab.loadDataWizard(''folder'');mobilab.gui;');
                uimenu(hImport,'Label','From DataRiver .bdf file','Callback','mobilab.loadDataWizard(''dr_bdf'');mobilab.gui;');
                
                uimenu(hFile,'Label','Load','Callback','mobilab.loadDataWizard(''mobi'');mobilab.gui;');
                uimenu(hFile,'Label','Save as','Callback','mobilab.saveAs;mobilab.gui;');
                uimenu(hFile,'Label','Close folder','Callback','delete(mobilab.allStreams);mobilab.gui;');
                
                uimenu(hFile,'Label','Exit MoBILAB','Callback','mobilab.applicationClose;');
                
                hEdit = uimenu('Label','Edit','Parent',figureHandle);
                uimenu(hEdit,'Label','Preferences','Callback','mobilab.setPreferences');
                
                hTools = uimenu('Label','Tools','Parent',figureHandle);
                uimenu(hTools,'Label','MultiStream browser','Callback','mobilab.msBrowser;');
                %uimenu(hTools,'Label','Copy and Import folders (batch mode)','Callback','mobilab.copyImportFolder;');
                %uimenu(hTools,'Label','Mocap workflow','Callback','mobilab.mocapWorkflow;');
                %uimenu(hTools,'Label','Mocap events editor','Callback','mobilab.eventsEditor;');
                
                hHelp = uimenu('Label','Help','Parent',figureHandle);
                uimenu(hHelp,'Label','Technical documentation','Callback','mobilab.onlineHelp;');
                uimenu(hHelp,'Label','GUI operation','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_toolbox_tutorial'']);');
                uimenu(hHelp,'Label','Scripting examples','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line'']);');
                uimenu(hHelp,'Label','Head model','Callback','mobilab.onlineHelp([mobilab.doc ''/headModel'']);');
                hHow2 = uimenu(hHelp,'Label','Howto');
                uimenu(hHow2,'Label','Export MoBILAB''s objects to EEGLAB','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#Step_6:_Exporting_results_to_EEGLAB'']);');
                uimenu(hHow2,'Label','Insert events','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#Step_5:_Computing_the_derivatives_on_the_PCA_projections'']);');
                uimenu(hHow2,'Label','Delete events','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#Step_5:_Computing_the_derivatives_on_the_PCA_projections'']);');
                uimenu(hHow2,'Label','Export to EEGLAB EEG data and events that are in different objects','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#How_to_export_to_EEGLAB_EEG_data_and_events_that_are_in_different_objects'']);');
                uimenu(hHow2,'Label','Delete objects from the command line','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#Delete_objects_from_the_command_line'']);');
                uimenu(hHow2,'Label','Transfer events between objects with different sampling rate','Callback','mobilab.onlineHelp([mobilab.doc ''/MoBILAB_from_the_command_line#Transfer_events_between_objects_with_different_sampling_rate'']);');
                
                
                %uimenu(hHelp,'Label','Authors','Callback','web([mobilab.doc ''/MoBILAB_from_the_command_line'']);');
                %uimenu(hHelp,'Label','Thanks to','Callback','web([mobilab.doc ''/MoBILAB_from_the_command_line'']);');
                uimenu(hHelp,'Label','License','Callback','mobilabLicense(mobilab);');
                
                % hcmenu = uicontextmenu('Parent',figureHandle);
                % uimenu(hcmenu, 'Label', 'Refresh', 'Callback', 'mobilab.refresh;');
                % set(figureHandle,'UIContextMenu',hcmenu);
            end
            
            set(figureHandle,'Color',obj.preferences.gui.backgroundColor);
            
            if isempty(obj.allStreams) || ~isvalid(obj.allStreams)
                % cleaning the form
                try delete(findobj(figureHandle,'Tag','mobilabTreeStructure'));end %#ok
                set(figureHandle,'Name','MoBILAB (Load some data to start)','Visible','on');
                delete(findobj(figureHandle,'type','axes'));
                drawnow;
                warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
                jFrame = get(handle(figureHandle),'JavaFrame');
                try
                    jRootPane = jFrame.fHG1Client.getWindow;
                catch
                    jRootPane = jFrame.fHG2Client.getWindow;
                end
                    
                obj.statusBar = com.mathworks.mwswing.MJStatusBar;
                javaObjectEDT(obj.statusBar);
                
                % Add a progress-bar to left side of standard MJStatusBar container
                obj.progressBar = javax.swing.JProgressBar;
                javaObjectEDT(obj.progressBar);
                set(handle(obj.progressBar), 'Minimum',0, 'Maximum',1, 'Value',0);
                obj.statusBar.add(obj.progressBar,'West');
                
                % Set this container as the figure's status-bar
                jRootPane.setStatusBar(obj.statusBar);
                
                obj.statusBar.setVisible(0);
                obj.statusBar.setText('');
                
                return
            end
            
            set(figureHandle,'Name','MoBILAB');
            
            switch lower(callback)
                case 'dispnode_callback'
                    funcHandle = @disp;
                case 'generatebatch_callback'
                    funcHandle = @generateBatch_Callback;
                case 'deletenode_callback'
                    funcHandle = @deleteNode_Callback;
                case 'addbrowserlist_callback'
                    funcHandle = @addBrowserList_Callback;
                case 'selectstream_callback'
                    funcHandle = @selectStream_Callback;
                case 'add2eventseditor2_callback'
                    funcHandle = @add2EventsEditor2_Callback;
                case 'add2browser_callback'
                    if isMatlab2014b
                        funcHandle = @MultiStreamBrowser2014b;
                    else
                        funcHandle = @MultiStreamBrowser;
                    end
                otherwise
                    funcHandle = @disp;
            end
            
            N = length(obj.allStreams.item);
            if isempty(obj.allStreams.gObj)
                [~,gObj] = obj.allStreams.viewLogicalStructure(callback,false);
            else
                gObj = obj.allStreams.gObj;
            end
            
            callbacks = cell(N,1);
            for it=1:N, callbacks{it} = {funcHandle,obj.allStreams.item{it}};end
            
            try
                
                % cleaning the form
                try delete(findobj(figureHandle,'Tag','mobilabTreeStructure'));end %#ok
                
                N = size(gObj.adjacencyMatrix,1);
                node = cell(N-1,1);
                nodeIDs = cell(N,1);
                contextMenuItems = cell(N-1,1);
                [~,nodeIDs{1}] = fileparts(obj.allStreams.mobiDataDirectory);
                for it=2:N, nodeIDs{it} = ['(' num2str(it-1) ') ' obj.allStreams.item{it-1}.name];end
                icon = fullfile(obj.path,'skin','mobilabObject2.png');
                root = javax.swing.tree.DefaultMutableTreeNode( nodeIDs{1} );
                javaObjectEDT(root);
                for it=1:N-1
                    node{it} = javax.swing.tree.DefaultMutableTreeNode(nodeIDs{it+1});
                    if strcmpi(callback,'add2Browser_Callback')
                        contextMenuItems{it} = javax.swing.JPopupMenu;
                        menuItem = javax.swing.JMenuItem('Plot in MS Browser');
                        javaObjectEDT(menuItem);
                        if isMatlab2014b
                            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@MultiStreamBrowser2014b; obj.allStreams.item{it}});
                        else
                            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@MultiStreamBrowser; obj.allStreams.item{it}});
                        end
                        contextMenuItems{it}.add(menuItem);
                    else
                        contextMenuItems{it} = obj.allStreams.item{it}.contextMenu;
                        javaObjectEDT(contextMenuItems{it});
                    end
                end
                ind = find(gObj.adjacencyMatrix(1,:));
                for jt=1:length(ind), root.add(node{ind(jt)-1});end
                for it=1:N-1
                    ind = getDescendants(gObj,it+1);
                    for jt=1:length(ind), try node{it}.add(node{ind(jt)-1});end;end %#ok
                end
                
                jTree = javax.swing.JTree(root);
                javaObjectEDT(jTree);
                jTree.setBackground(java.awt.Color(obj.preferences.gui.backgroundColor(1),...
                    obj.preferences.gui.backgroundColor(2),obj.preferences.gui.backgroundColor(3)));
                jTree.setForeground(java.awt.Color(obj.preferences.gui.backgroundColor(1),...
                    obj.preferences.gui.backgroundColor(2),obj.preferences.gui.backgroundColor(3)));
                jImageIcon = javax.swing.ImageIcon(icon);
                renderer = javax.swing.tree.DefaultTreeCellRenderer;
                renderer.setBackground(java.awt.Color(obj.preferences.gui.backgroundColor(1),...
                    obj.preferences.gui.backgroundColor(2),obj.preferences.gui.backgroundColor(3)));
                renderer.setBackgroundNonSelectionColor(java.awt.Color(obj.preferences.gui.backgroundColor(1),...
                    obj.preferences.gui.backgroundColor(2),obj.preferences.gui.backgroundColor(3)));
                renderer.setBackgroundSelectionColor(java.awt.Color(78/256,80/256,76/256))
                renderer.setTextNonSelectionColor(java.awt.Color(obj.preferences.gui.fontColor(1),...
                    obj.preferences.gui.fontColor(2),obj.preferences.gui.fontColor(3)))
                renderer.setIcon(jImageIcon);
                renderer.setClosedIcon(jImageIcon);
                renderer.setOpenIcon(jImageIcon);
                renderer.setLeafIcon(jImageIcon);
                javax.swing.ToolTipManager.sharedInstance().registerComponent(jTree);
                jTree.setCellRenderer(renderer);
                %set(jTree,'userData',callbacks);
                set(figureHandle,'userData',callbacks)
                jScrollPane = com.mathworks.mwswing.MJScrollPane(jTree);
                javaObjectEDT(jScrollPane)
                jScrollPane.setBackground(java.awt.Color(obj.preferences.gui.backgroundColor(1),...
                    obj.preferences.gui.backgroundColor(2),obj.preferences.gui.backgroundColor(3)));
                
                position = get(figureHandle,'Position');
                [~,hc ] = javacomponent(jScrollPane,[2 1.5 ,position(3:4)],figureHandle);
                set(hc,'Units','Normalized','Tag','mobilabTreeStructure','Position',[0 0 1 1]);
                
                rootMenu = javax.swing.JPopupMenu;
                javaObjectEDT(rootMenu)
                rootMenuItem = javax.swing.JMenuItem('Annotation');
                javaObjectEDT(rootMenuItem);
                rootMenu.add(rootMenuItem);
                set(handle(rootMenuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj.allStreams});
                
                guiMenu = javax.swing.JPopupMenu;
                javaObjectEDT(guiMenu);
                guiMenuItem = javax.swing.JMenuItem('Refresh');
                javaObjectEDT(guiMenuItem);
                guiMenu.add(guiMenuItem);
                set(handle(guiMenuItem,'CallbackProperties'), 'ActionPerformedCallback', {@refresh_Callback,obj});
                
                % I'll work around this warning in the future
                warning off %#ok
                set(jTree, 'MousePressedCallback', {@mousePressedCallback,cat(1,{guiMenu;rootMenu},contextMenuItems)});
                warning on; %#ok
                set(figureHandle,'Visible','on');
                
                drawnow;
                warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
                jFrame = get(handle(figureHandle),'JavaFrame');
                try
                    jRootPane = jFrame.fHG1Client.getWindow;
                catch
                    jRootPane = jFrame.fHG2Client.getWindow;
                end
                javaObjectEDT(jRootPane);
                obj.statusBar = com.mathworks.mwswing.MJStatusBar;
                javaObjectEDT(obj.statusBar)
                
                % Add a progress-bar to left side of standard MJStatusBar container
                obj.progressBar = javax.swing.JProgressBar;
                javaObjectEDT(obj.progressBar)
                set(obj.progressBar, 'Minimum',0, 'Maximum',1, 'Value',0);
                obj.statusBar.add(obj.progressBar,'West');
                
                % Set this container as the figure's status-bar
                jRootPane.setStatusBar(obj.statusBar);
                
                obj.statusBar.setText('');
                obj.statusBar.setVisible(0);
                
            catch ME
                ME.rethrow;
            end
        end
        %%
        function msBrowserHandle = msBrowser(obj)
            if isempty(obj.allStreams) || ~isvalid(obj.allStreams), error('Load some data first');end
            if isMatlab2014b
                msBrowserHandle = MultiStreamBrowser2014b(obj);
            else
                msBrowserHandle = MultiStreamBrowser(obj);
            end
        end
        %%
        function copyImportFolder(obj,source,destination)
            if nargin < 2
                h = CopyImport(obj);
                uiwait(h);
                userData = get(h,'UserData');
                delete(h);
                if isempty(userData), return;end
                source      = userData.source;
                destination = userData.destination;
                drawnow;
            end
            
            if ~exist(source,'file'), error('Source folde does not exist.');end
            if ~exist(destination,'dir'), 
                disp(['Creating directory: ' destination])
                mkdir(destination);
            end
            disp('Copying...')
            copyfolder(source,destination);
            %--
            list = dir(destination);
            isFile = logical(~cell2mat({list.isdir}));
            names = {list.name};
            filesnames = names(isFile);
            [~,loc] = min(cellfun(@length,filesnames));
            [~,n,e] = fileparts(filesnames{loc});
            folderName = [n '_MoBI'];
            mobiDataDirectory = fullfile(destination,folderName);
            %--
            obj.allStreams = dataSourceFromFolder(destination,mobiDataDirectory);
            if obj.isGuiActive, obj.gui;end
        end
        %%
        function fHandle = mocapWorkflow(obj)
            if isempty(obj.allStreams) || ~isvalid(obj.allStreams), error('Load some data first');end
            fHandle = MocapPipeline(obj);
        end
        %%
        function eeHandle = eventsEditor(obj)
            if isempty(obj.allStreams) || ~isvalid(obj.allStreams), errord('Load some data first');end
            isPCA = false;
            for it=1:length(obj.allStreams.item)
                if isa(obj.allStreams.item{it},'vectorMeasureInSegments') || isa(obj.allStreams.item{it},'projectedMocap')
                    isPCA = true;
                    break;
                end
            end
            if ~isPCA, error('Run PCA on mocap data first.');end
            eeHandle = EventsEditor(obj);
        end
        %%
        function initApplication(obj)
            if exist(fullfile(getHomeDir,'.mobilab.mat'),'file')
                load(fullfile(getHomeDir,'.mobilab.mat'));
                if ~exist('configuration','var')
                    delete(fullfile(getHomeDir,'.mobilab.mat'));
                    obj.initApplication;
                else
                    obj.preferences = configuration; %#ok
                end                
                if ~isfield(obj.preferences.mocap,'bodyModel')
                    obj.preferences.mocap.bodyModel = fullfile(obj.path,'data','KinectBodyModel.mat');
                end
            else
                obj.preferences.gui.backgroundColor = [0.93 0.96 1]; % default eeglab's color: [0.66 0.76 1]
                obj.preferences.gui.buttonColor = [1 1 1];
                obj.preferences.gui.fontColor = [0 0 0.4];
                obj.preferences.mocap.interpolation = 'pchip';
                obj.preferences.mocap.smoothing = 'sgolay';
                obj.preferences.mocap.lowpassCutoff = 6;
                obj.preferences.mocap.derivationOrder = 3;
                obj.preferences.mocap.stickFigure = fullfile(obj.path,'data','sccnOnePersonStickFigure.mat');
                obj.preferences.mocap.bodyModel = fullfile(obj.path,'data','KinectBodyModel.mat');
                obj.preferences.eeg.resampleMethod = 'linear';
                obj.preferences.eeg.filterType = 'bandpass';
                obj.preferences.eeg.cutoff = [1 200];
                obj.preferences.eeg.headModel = fullfile(obj.path,'data','head_modelColin27_4825.mat');
                obj.preferences.tmpDirectory = tempdir;
                
                answer = [];
                options.Resize='on';
                options.WindowStyle='normal';
                while isempty(answer)
                    answer = inputdlg({'Name' 'Organization' 'email'},'Please enter your contact info',1,{getUserName 'SCCN' [getUserName '@sccn.ucsd.edu']},options);
                end
                obj.preferences.username = answer{1};
                obj.preferences.organization = answer{2};
                obj.preferences.email = answer{3};
                configuration = obj.preferences; %#ok
                save(fullfile(getHomeDir,'.mobilab.mat'),'configuration');
            end
        end
        %%
        function initStatusbar(obj,mn,mx,msg)
            if nargin < 1, mn = 0;end
            if nargin < 2, mx = 1;end
            if nargin < 3, msg = '';end
            if ~obj.isGuiActive, disp(msg);return;end
            set(obj.progressBar, 'Minimum',mn, 'Maximum',mx, 'Value',0);
            obj.statusBar.setText(msg);
            obj.statusBar.setVisible(1);
            drawnow;
        end
        %%
        function statusbar(obj,value,msg)
            if ~obj.isGuiActive, return;end
            if nargin < 2, value = inf;end
            if nargin == 3, obj.statusBar.setText(msg);end
            set(obj.progressBar,'Value',value);
            if value >= obj.progressBar.getMaximum
                obj.statusBar.setText('Done!!!');
                pause(1);
                obj.statusBar.setVisible(0);
                return;
            end
        end
        %%
        function lockGui(obj,msg)
            persistent flag;
            persistent hwait;
            if nargin < 2, msg = 'Do not press Ctrl+C ... ';end
            if isempty(flag), flag = true;end
            if isMatlab2014b, return;end
            if ~obj.isGuiActive
                if flag, disp(msg);else disp('Done!');end
                flag = ~flag;
                return;
            end
            
            if flag
                try
                    [~,figureHandle] = obj.isGuiActive();
                    isMatlab2014b
                    warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
                    jFrame = get(handle(figureHandle),'JavaFrame');
                    try
                        jWindow = jFrame.fHG1Client.getWindow;
                    catch
                        jWindow = jFrame.fHG2Client.getWindow;
                    end
                    hwait = com.mathworks.mlwidgets.dialog.ProgressBarDialog.createHeavyweightInternalProgressBar(jWindow,' ',[]);
                    hwait.setProgressStatusLabel(msg);
                    hwait.setSpinnerVisible(false);
                    hwait.setCancelButtonVisible(false);
                    hwait.setVisible(true);
                    hwait.setCircularProgressBar(true);
                    drawnow;
                end
            else
                try hwait.dispose;end %#ok
            end
            flag = ~flag;
        end
        %%
        function h = waitCircle(obj,msg)
            if nargin < 2, msg = '';end
            iconClassName = 'com.mathworks.widgets.BusyAffordance$AffordanceSize';
            iconSizeEnums = javaMethod('values', iconClassName);
            size_32x32 = iconSizeEnums(2);
            jObj = com.mathworks.widgets.BusyAffordance(size_32x32,msg);
            jObj.setPaintsWhenStopped(true)
            h = figure('MenuBar','none','ToolBar','none','NumberTitle','off','Position',[1 1 80 80],'Color',obj.preferences.gui.backgroundColor,'Visible','off');
            
            eeglabFigure = findobj(0,'Tag','EEGLAB');
            if isempty(eeglabFigure)
                movegui(h,'center')
            else
                position = get(eeglabFigure,'Position');
                movegui(h,(position(1:2)+position(3:4)))
            end
            pause(0.125);
            set(h,'Units','Normalized','Resize','off','Visible','on');
            javacomponent(jObj.getComponent,[1 1 80 80],h);
            jObj.start
            drawnow
        end
        %%
        function saveAs(obj,folder)
            if isempty(obj.allStreams) || ~isvalid(obj.allStreams), return;end
            if nargin < 2, folder = '';end
            if ~exist(folder,'dir')
                h = SaveAs(obj);
                uiwait(h);
                userData = get(h,'UserData');
                delete(h);
                if isempty(userData), return;end
                mobiDataDirectory = userData.mobiDataDirectory;
            else
                mobiDataDirectory = folder;
            end
            obj.allStreams.save(mobiDataDirectory);
        end
        %%
        function loadDataWizard(obj,format)
            switch lower(format)
                case 'file',   guiFun = @ImportFile;
                case 'folder', guiFun = @ImportFolder;
                case 'dr_bdf', guiFun = @ImportFromDatariverBDF;
                case 'mobi'
                    mobiDataDirectory = uigetdir2('Select the _MoBI folder');
                    if isnumeric(mobiDataDirectory), return;end
                    if ~exist(mobiDataDirectory,'dir'), return;end
                    suffix = '_MoBI';
                    suffixLength = length(suffix);
                    if ~strcmp(mobiDataDirectory(end-suffixLength+1:end),suffix)
                        errordlg('This is not a valid _MoBI folder');
                        return
                    end
                    obj.allStreams = dataSourceMoBI(mobiDataDirectory);
                    if obj.isGuiActive, obj.gui;end
                    return;
            end
            
            try
                h = guiFun(obj);
                uiwait(h);
                userData = get(h,'UserData');
                delete(h);
                drawnow;
                if isempty(userData), return;end
                source = userData.source;
                mobiDataDirectory = userData.mobiDataDirectory;
                
                [~,~,ext] = fileparts(source);
                if isempty(ext) && exist(source,'dir'), ext = 'dir';end
                
                switch ext
                    case '.xdf',  importFun = @dataSourceXDF; importFunStr = 'dataSourceXDF';
                    case '.xdfz', importFun = @dataSourceXDF; importFunStr = 'dataSourceXDF';
                    case '.set',  importFun = @dataSourceSET; importFunStr = 'dataSourceSET';
                    case '.drf',  importFun = @dataSourceDRF; importFunStr = 'dataSourceDRF';
                    case '.bdf',  importFun = @dataSourceBDF; importFunStr = 'dataSourceBDF';
                    case 'dir',   importFun = @dataSourceFromFolder; importFunStr = 'dataSourceFromFolder';
                    otherwise,    error('Unknown format.')
                end
                
                if strcmpi(ext,'.bdf')
                    disp('Running:');
                    disp(['  mobilab.allStreams = ' importFunStr '( ''' source ''' , ''' mobiDataDirectory ''', configList );' ]);
                    
                    configList = userData.configList;
                    obj.allStreams = importFun(source,mobiDataDirectory,configList);
                    return;
                end
                
                disp('Running:');
                disp(['  mobilab.allStreams = ' importFunStr '( ''' source ''' , ''' mobiDataDirectory ''');' ]);
                
                obj.allStreams = importFun(source,mobiDataDirectory);
                if obj.isGuiActive, obj.gui;end
            catch ME
                errordlg(ME.message);
            end
        end
        %%
        function refresh(obj,~)
            [isActive,figureHandle] = obj.isGuiActive;
            if isActive, return;end
            close(figureHandle);
            obj.gui;
        end
    end
    methods(Static)
        %%
        function [isActive,h] = isGuiActive
            isActive = true;
            h = findobj(0, 'Tag','mobilabApplicationGUI');
            if isempty(h), isActive = false;end
        end
    end
end

%% Set the mouse-press callback
function mousePressedCallback(hTree, eventData, jmenu)
persistent count  t,
th = 0.006;
% Get the clicked node
clickX = eventData.getX;
clickY = eventData.getY;
jtree = eventData.getSource;
treePath = jtree.getPathForLocation(clickX, clickY);

if isempty(treePath)
    if ~eventData.isMetaDown  , return;end
    jmenu{1}.show(jtree, clickX, clickY);
    jmenu{1}.repaint;
    return;
else
    jmenu(1) = [];
end
node = treePath.getLastPathComponent;
nodeName = char(node);
loc = find(nodeName == '(' | nodeName == ')');
if isempty(loc)
    if ~eventData.isMetaDown  , return;end
    jmenu{1}.show(jtree, clickX, clickY);
    jmenu{1}.repaint;
    return;
end
index = str2double(nodeName(loc(1)+1:loc(2)-1));

% right-click is like a Meta-button
if eventData.isMetaDown  
    
    % Display the (possibly-modified) context menu
    jmenu{index+1}.show(jtree, clickX, clickY);
    jmenu{index+1}.repaint;
else
    if isempty(count), count = 1;t=now;end
    if count == 2 && (now - t)*1e3 < th
        %callbacks = get(hTree,'userData');
        hFigure = findobj('Name','MoBILAB');
        callbacks = get(hFigure,'userData');
        if isempty(callbacks), return;end
        funcHandle = callbacks{index}{1};
        arg = callbacks{index}{2};
        funcHandle(arg);
        count = 1;
    elseif count > 2
        count = 1;
        t = now;
    else
        count = count+eventData.getClickCount;
        t = now;
    end
end
end

%%
function tree = getDirectoryTree(rootDir)
thisTree = dir(rootDir);
I = cell2mat({thisTree.isdir});
thisTree = {thisTree.name};
thisTree = thisTree(I);
thisTree(strcmp(thisTree,'private')) = [];
rmThis = false(length(thisTree),1);
n = length(thisTree);
n(n>2) = 3;
for it=1:n, if thisTree{it}(1) == '.', rmThis(it) = true;end;end
thisTree(rmThis) = [];
tree = {rootDir};
for it=1:length(thisTree)
    subTree = getDirectoryTree([rootDir filesep thisTree{it}]);
    tree = cat(1,tree,subTree);
end
ind = ~cellfun(@isempty,strfind(tree,'old'));
if any(ind) && length(tree) > 1
    old = tree(ind);
    tree(ind) = [];
    tree = cat(1,tree,old);
end
end

%%
function addDependency2Path(dependencyTree)
for it=1:length(dependencyTree)
    list = dir(dependencyTree{it});
    I = ~cell2mat({list.isdir});
    if any(I)
        files = pickfiles(dependencyTree{it},'.m','.m','Content');
        if ~isempty(files)
            [~,filename] = fileparts(deblank(files(1,:)));
            if ~exist(filename,'file')
                addpath(dependencyTree{it});
            else
                desc1 = dir(which(filename));
                desc2 = dir(deblank(files(1,:)));
                if isempty(desc1), addpath(dependencyTree{it});
                elseif desc2.datenum > desc1.datenum, addpath(dependencyTree{it});
                end
            end
        end
    end
end
end
