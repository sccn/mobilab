% Definition of the class dataSource. This is an abstract class that serves
% as a base for classes that implement interfaces between data acquisition 
% systems or intermediate data formats and MoBILAB toolbox.
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Apr-2011

%%
classdef dataSource < handle
    properties(SetAccess=protected) 
        sessionUUID                   % Universal unique identifier that links all the data sets
                                      % belonging to the same session. A session is defined as a
                                      % raw file or collection of files recorded the same day and
                                      % its derived processed files. To guarantee that files within
                                      % the same session are traceable, the  sessionUUID code is 
                                      % attached at the end of each file name.
                                      % 
                                      % Example: 
                                      %         dataset_1: filename_1_sessionUUID.hdr
                                      %                    filename_1_sessionUUID.bin
                                      %         dataset_2: filename_2_sessionUUID.hdr
                                      %                    filename_2_sessionUUID.bin
                                      %         dataset_n: filename_n_sessionUUID.hdr
                                      %                    filename_n_sessionUUID.bin
                                      
    end
    properties(SetObservable)
        item                          % Cell array (list) of objects (handles), where each object represents
                                      % one single data set. The objects connect to binary and header files
                                      % in the mobiDataDirectory and come up to live as functional entities that
                                      % "know" what to do with the data sets they handle. 
    end
    properties(GetAccess=public, SetAccess=public)
        mobiDataDirectory;            % Path to the directory where the collection of files of one session are
                                      % stored.
                                      
        container                     % The term container is used to refer to an object that can perform certain
                                      % operations on the contained object but not the other way around. In this 
                                      % case, for a dataSource object the container is an object from the class 
                                      % mobilabApplication. The role of mobilabApplication is controlling the GUI,
                                      % but because GUI needs change rapidly, a different class can be implemented
                                      % to satisfy the new requirements in the front-end and still serve as container
                                      % to the dataSource. This is a way of separating GUI programming from the rest
                                      % of the application that may be more stable.
                                      
        notes = mobiAnnotator;
    end
    properties(Access = protected, Hidden), hashTable = [];end
    properties(GetAccess=public, SetAccess=protected, Hidden)
        gObj
        listenerHandle
        logicalStructure
    end
    methods
        %%
        function obj = dataSource(varargin)
            % Creates a dataSource object. As this is an abstract class it 
            % cannot be instantiated on its own. Therefore, this constructor
            % must be call only within a child class constructor.
            % 
            % Input arguments:
            %       mobiDataDirectory: path to the directory where the collection
            %                          of files are or will be stored.
            %       sessionUUID:       string specifying the universal unique identifier
            %                          common to all the files of the same session.
            %       container:         this is usually a mobilabApplication object that
            %                          connects the GUI with the low-level functions that
            %                          each item in the data source may implement.  
            %                             
            % Output argument:           
            %       obj:               dataSource (handle)
            % 
            % Usage:
            %       obj@dataSource( mobiDataDirectory, sessionUUID, mobilab);

            if nargin < 1, varargin{1} = '';end   % mobiDataDirectory
            if nargin < 2, varargin{2} = '';end   % sessionUUID
            if nargin < 3, varargin{3} = '';end   % container
            obj.mobiDataDirectory = varargin{1};
            obj.sessionUUID = varargin{2};
            if ~isa(varargin{3},'mobilabApplication')
                try containerObj = evalin('base','mobilab');
                    if ~isempty(containerObj.allStreams) && isvalid(containerObj.allStreams), delete(containerObj.allStreams);end
                catch
                    containerObj = mobilabApplication(obj);
                    assignin('base','mobilab',containerObj)
                end
            else containerObj = varargin{3};
            end
            obj.container = containerObj;
            notesFile = [obj.mobiDataDirectory filesep 'notes_' obj.sessionUUID '.txt'];
            if exist(notesFile,'file')
                txt = textfile2cell(notesFile);
                obj.notes = mobiAnnotator(obj,txt);
            else obj.notes = mobiAnnotator(obj);
            end
            obj.item = {};
            obj.logicalStructure = 0;
            obj.listenerHandle = obj.addlistener('item','PostSet',@dataSource.updateTree);
        end
        %%
        function item = get.item(obj), item = obj.item;end
        function mobiDataDirectory = get.mobiDataDirectory(obj), mobiDataDirectory = obj.mobiDataDirectory;end
        function set.mobiDataDirectory(obj,mobiDataDirectory), obj.mobiDataDirectory = mobiDataDirectory;end
        function gObj = get.gObj(obj)
            if isempty(obj.gObj), obj.gObj = graphCoreObject(obj.logicalStructure);end
            gObj = obj.gObj;
        end
        function delete(obj)
            try obj.save;
                N = length(obj.item);
                for it=1:N, delete(obj.item{it});end
            catch ME, disp(ME.message);
            end
        end
        %%
        function cobj = addItem(obj,header)
            % Adds an object to the list of items "item". The method also updates
            % the logical connections between the new object and the rest of the tree.
            % 
            % Input arguments:
            %       header: header file (string)
            % 
            % Output arguments:
            %       obj: handle to the new object
            % 
            % Usage: 
            %       cobj = obj.addItem( header );
            
            if nargin < 1, error('Not enough input arguments.');end
            try load(header,'-mat','class');
                constructorHandle = eval(['@' class]); %#ok
                cobj = constructorHandle(header);
            catch ME
                warning(ME.message)
                if exist('class','var')
                    switch class
                        case 'projectedMocap', class = 'pcaMocap';
                        case 'segmentedMocap', class = 'mocap';
                        otherwise, class = 'dataStream';
                    end
                else class = 'dataStream';
                end
                save(header,'-mat','-append','class');
                constructorHandle = eval(['@' class]);
                cobj = constructorHandle(header);
            end
            disp(['Adding object: ' cobj.name]);
            obj.item{end+1} = cobj;
            cobj.container = obj;
        end
        %%
        function deleteItem(obj,itemIndex)
            % Deletes one or more objects from the list of items. It also removes
            % the underlying header and binary files from the mobiDataDirectory.
            % Then, the logical connections between the remaining objects are updated.
            % It is not possible to remove intermediate objects without removing its
            % children. Objects containing raw data cannot be erased.
            % 
            % Input arguments:
            %       indices: indices in the cell array "item" of the objects to remove 

            if nargin < 2, error('Not enough input arguments.');end
            N = length(itemIndex);
            if N > 1 && isnumeric(itemIndex)
                if any(itemIndex > length(obj.item)), return;end
                uuid = cell(N,1);
                for it=1:N, uuid{it} = obj.item{itemIndex(it)}.uuid;end
                for it=1:N, deleteItem(obj,uuid{it});end
                return
            end
            if ischar(itemIndex), itemIndex = obj.findItem(itemIndex);end
            if itemIndex > length(obj.item), return;end
            delList = getIndices4aBranch(obj,itemIndex);
            
            for it=1:length(delList),
                if ~obj.item{delList(it)}.writable, error('MoBILAB:attempt_to_delete_read_only_object','Cannot delete files containing raw data.');end
                disp(['Removing object: ' obj.item{delList(it)}.name]);
                bin2delete = obj.item{delList(it)}.binFile;
                hdr2delete = obj.item{delList(it)}.header;
                delete(obj.item{delList(it)});
                java.io.File(bin2delete).delete();
                java.io.File(hdr2delete).delete();
            end
            obj.item(delList) = [];
        end
        %%
        function save(obj,newDataSourceLocation)
            % Saves the content of the mobiDataDirectory in a new directory
            % and connects the dataSource to that directory. The user experiences
            % no change but all the objects are destroyed and created again 
            % pointing to files in the new mobiDataDirectory.
            
            if nargin < 2, newDataSourceLocation = obj.mobiDataDirectory;end
            if isempty(newDataSourceLocation), error('prog:input','You must provide a filename where save the datasource.');end
            if ~exist(newDataSourceLocation,'dir'), mkdir(newDataSourceLocation);end
            
            tmpMobiDataDirectory = obj.mobiDataDirectory;
            if newDataSourceLocation(end)==filesep, newDataSourceLocation(end)=[];end
            if obj.mobiDataDirectory(end)==filesep, tmpMobiDataDirectory(end)=[];end
            
            N = length(obj.item);
            copyFlag = ~strcmp(newDataSourceLocation,tmpMobiDataDirectory);
            if copyFlag, obj.container.initStatusbar(1,N,'Copying files...');end
            for it=1:N
                if isvalid(obj.item{it})
                    if copyFlag
                        obj.item{it}.disconnect;
                        [~,filename] = fileparts(obj.item{it}.header);
                        copyfile(obj.item{it}.header,newDataSourceLocation);
                        copyfile(obj.item{it}.binFile,newDataSourceLocation);
                        obj.item{it}.header  = fullfile(newDataSourceLocation,[filename '.hdr']);
                        obj.item{it}.binFile = fullfile(newDataSourceLocation,[filename '.bin']);
                        saveProperty(obj.item{it},'header',obj.item{it}.header);
                        saveProperty(obj.item{it},'binFile',obj.item{it}.binFile);
                        obj.item{it}.connect;
                        obj.container.statusbar(it);
                    end
                end
            end
            cell2textfile([newDataSourceLocation filesep 'notes_' obj.sessionUUID '.txt'],obj.notes.text)
            obj.mobiDataDirectory = newDataSourceLocation;
        end
        %%
        function indices = getDescendants(obj,index)
            % Returns indices of item's direct descendants.
            % 
            % Input arguments:
            %       index:       index of an object in "item" list
            % 
            % Output arguments:
            %       indices:     indices of the direct descendants of item{ index }

            if nargin < 2, error('Not enough input arguments.');end
            indices = obj.gObj.getDescendants(index+1)-1;
            indices(indices==0) = [];
        end
        function indices = getAncestors(obj,index)
            % Returns indices of item's ancestors at all previous levels.
            % 
            % Input arguments:
            %       index:       index of an object in "item" list
            % 
            % Output arguments:
            %       indices:     indices of all ancestors of item{ index }
            
            if nargin < 2, error('Not enough input arguments.');end
            indices = obj.gObj.getAncestors(index+1)-1;
            indices(indices==0) = [];
        end
        %%
        function itemIndex = findItem(obj,uuid)
            % Returns the index in "item" list corresponding a uuid.
            % 
            % Input argument:
            %       uuid:      universal unique identifier (string)
            % 
            % Output argument:
            %        index:    index of the object whose uuid matched
            
            if nargin < 2, error('Not enough input arguments.');end
            if ~iscellstr(uuid) && ~ischar(uuid), error('Input must be a string or a cell array of strings.');end
            if ~iscellstr(uuid), uuid = {uuid};end
            N = length(uuid);
            itemIndex = zeros(N,1);
            for it=1:N,
                ind = find(ismember(obj.hashTable,uuid{it}));
                if ~isempty(ind), itemIndex(it) = ind;end
            end
        end
        function [index,uuid] = getItemIndexFromItemName(obj,name)
            % Returns indices of object whose name match the input string,
            % if no match is found it returns 0.
            % 
            % Input arguments:
            %       objectName:  object's name (string)
            % 
            % Output arguments:
            %        indices:    indices of objects whose name matched the input string
            %        uuid:       correspondent uuids, if more than one is returned uuid
            %                    is a cell array, otherwise is a string
            %
            % Usage:
            %        % Plots all objects whose name is 'filt_eeg'
            %        objectName = 'filt_eeg'
            %        indices = mobilab.allStreams.getItemIndexFromItemName( objectName );
            %        for it=1:length(indices), plot( mobilab.allStreams.item{ indices(it) } );end 
    
            if nargin < 2, error('Not enough input arguments.');end   
            N = length(obj.item);
            objName = cell(N,1);
            for it=1:N, objName{it} = obj.item{it}.name;end
            index = find(ismember(objName,name));
            if nargout > 1
                uuid = cell(length(index),1);
                for it=1:length(index), uuid{it} = obj.item{index(it)}.uuid;end
                if length(uuid) == 1, uuid = uuid{1};end
            end
        end
        function [index,uuid] = getItemIndexFromItemNameSimilarTo(obj,name)
            % Returns indices of objects whose name contain the input string,
            % if no match is found it returns 0.
            % 
            % Input arguments:
            %       objectName: object's name (string)
            % 
            % Output arguments:
            %       indices:    indices of objects whose name include the input string
            %       uuid:       correspondent uuids, if more than one is returned uuid
            %                   is a cell array, otherwise is a string
            %
            % Usage:
            %        % Plots all objects whose name is a superset of '_eeg'
            %        objectName = 'eeg'
            %        indices = mobilab.allStreams.getItemIndexFromItemNameSimilarTo( objectName );
            %        for it=1:length(indices), plot( mobilab.allStreams.item{ indices(it) } );end 
            
            if nargin < 2, error('Not enough input arguments.');end   
            N = length(obj.item);
            if ~N, return;end
            index = zeros(N,1);
            for it=1:N, if ~isempty(strfind(obj.item{it}.name,name)), index(it) = it;end;end
            index(index==0) = [];
            if nargout > 1
                uuid = cell(length(index),1);
                for it=1:length(index), uuid{it} = obj.item{index(it)}.uuid;end
                if length(uuid) == 1, uuid = uuid{1};end
            end
        end
        %%
        function [index,uuid] = getItemIndexFromItemClass(obj,itemClass)
            % Returns indices of object whose class match the input string,
            % if no match is found it returns 0.
            % 
            % Input arguments:
            %       objectType: object's class name (string)
            %                    
            % Output arguments: 
            %       indices:    indices of objects from the class objectClass
            %       uuid:       correspondent uuids, if more than one is returned
            %                   uuid is a cell array, otherwise is a string
            %
            % Usage:
            %        % Plots all objects from the class mocap
            %        itemClass = 'mocap'
            %        indices   = mobilab.allStreams.getItemIndexFromItemClass( itemClass );
            %        for it=1:length(indices), plot( mobilab.allStreams.item{ indices(it) } );end 
            
            if nargin < 2, error('Not enough input arguments.');end    
            N = length(obj.item);
            if ~N, return;end
            objClass = cell(N,1);
            for it=1:N, objClass{it} = class(obj.item{it});end
            index = find(ismember(objClass,itemClass));
            if nargout > 1
                uuid = cell(length(index),1);
                for it=1:length(index), uuid{it} = obj.item{index(it)}.uuid;end
                if length(uuid) == 1, uuid = uuid{1};end
            end
        end
        %%
        function EEG = export2eeglab(obj,dataObjIndex,eventObjIndex,newEEGfile,updateGui)
            % Export an EEG structure combining multiple streams and event markers. This 
            % method takes care of aligning and re-sampling objects with different sampling
            % rates. The first argument is a vector with the indices of the objects whose
            % data will be concatenated (by the channel dimension) one after the other to 
            % make EEG.data. The second argument is a vector with the indices of the objects
            % whose event markers will be used to populate EEG.event.
            %
            % Input arguments:
            %       indicesData:    indices of objects whose channels will be concatenated and
            %                       used to form EEG.data
            %       indicesMarkers: induces of objects whose event markers will be used to populate
            %                       EEG.event
            %                       
            % Output arguments:     
            %       EEG:            EEG EEGLAB's data structure
            %
            % Usage:
            %       % index_eeg_object:    index of the object containing EEG data
            %       % index_marker_object: index of the object containing the event markers
            %
            %       EEG = mobilab.allStreams.export2eeglab( index_eeg_object, index_marker_object);

            eeglab_options;
            if ~option_savetwofiles 
                errordlg('Change EEGLAB preferences to save two files - Mobilab requires it');
            end
                        if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, eventObjIndex = dataObjIndex;end
            if nargin < 4, newEEGfile = [];end
            if nargin < 5, updateGui = true;end
            
            I = false(length(dataObjIndex),1);
            for k=1:length(dataObjIndex), I(k) = obj.item{dataObjIndex(k)}.isMemoryMappingActive;end
            dataObjIndex = dataObjIndex(I);
            if isempty(dataObjIndex), return;end
            loc = (1:length(dataObjIndex))';
            for k=1:length(dataObjIndex)
                if ~isempty(obj.item{dataObjIndex(k)}.label{1}) && ~isempty(strfind(lower(obj.item{dataObjIndex(k)}.label{1}),'unknown'))
                    loc = circshift(loc,1);
                end
            end
            dataObjIndex = dataObjIndex(loc);
            eventObjIndex = unique([eventObjIndex(:)' dataObjIndex(:)']);
            
            if isempty(newEEGfile), newEEGfile = [obj.mobiDataDirectory filesep obj.item{dataObjIndex(1)}.name '.set'];end
            if ~ischar(newEEGfile), error('Third argument must be a char!!!');end
            
            if exist(newEEGfile,'file') && obj.container.isGuiActive && nargout < 1
                choice = questdlg(sprintf(['The file ' newEEGfile ' already exist.\nWould you like to overwrite it?']),'Warning!!!','Yes','No','No');
                if ~strcmp(choice,'Yes')
                    [FileName,PathName] = uiputfile2({'*.set','EEGLAB (.set)'},'Select a location for the EEGLAB .set file');
                    if any([isnumeric(FileName) isnumeric(PathName)]), return;end
                    newEEGfile = fullfile(PathName,FileName);
                end
            end
            [path,name] = fileparts(newEEGfile);
            
            EEG = eeg_emptyset;
            EEG.filepath = path;
            EEG.filename = name;
            EEG.setname = name;
            EEG.data = [path filesep name '.fdt'];
            [EEG.times, Ntimepoints,Nchannels,labels,streamObjList,type,latency,hedTag] = alignStreams(EEG,obj.item(dataObjIndex),obj.item(eventObjIndex));
            EEG.srate = streamObjList{1}.samplingRate;
            EEG.nbchan = Nchannels;
            EEG.pnts = Ntimepoints;
            EEG.times = 1000*EEG.times;% from seconds to milliseconds
            EEG.xmin  = EEG.times(1);
            EEG.xmax  = EEG.times(end);
            EEG.trials = 1;
            
            if isa(streamObjList{1},'eeg')
                EEG.chaninfo.nosedir = '+X';
                if isfield(streamObjList{1}.hardwareMetaData,'desc'), EEG.etc.desc = streamObjList{1}.hardwareMetaData.desc;end
                if ~isempty(streamObjList{1}.fiducials), EEG.etc.fiducials = streamObjList{1}.fiducials;end
            end
            chanlocs = repmat(struct('labels',[],'type',[],'X',[],'Y',[],'Z',[],'radius',[],'theta',[]),EEG.nbchan,1);
            for k=1:EEG.nbchan, chanlocs(k).labels= labels{k};end
            
            locChannels = 1:streamObjList{1}.numberOfChannels;
            for k=1:length(dataObjIndex)
                if k > 1, locChannels = (1:streamObjList{k}.numberOfChannels) + streamObjList{k-1}.numberOfChannels;end
                switch class(streamObjList{k})
                    case 'eeg'
                        if isfield(streamObjList{k}.hardwareMetaData,'desc')
                            try
                                for jt=1:streamObjList{k}.numberOfChannels, chanlocs(jt).type = streamObjList{k}.hardwareMetaData.desc.channels.channel{jt}.type;end
                            catch
                                for jt=1:streamObjList{k}.numberOfChannels, chanlocs(jt).type = streamObjList{k}.hardwareMetaData.desc.channels.type;end
                            end
                        else
                            for jt=1:streamObjList{k}.numberOfChannels, chanlocs(jt).type = 'EEG';end
                        end
                        if ~isempty(streamObjList{k}.channelSpace)
                            for jt=1:streamObjList{k}.numberOfChannels
                                chanlocs(locChannels(jt)).X = streamObjList{k}.channelSpace(jt,2);
                                chanlocs(locChannels(jt)).Y = streamObjList{k}.channelSpace(jt,1);
                                chanlocs(locChannels(jt)).Z = streamObjList{k}.channelSpace(jt,3);
                                [chanlocs(locChannels(jt)).theta, chanlocs(locChannels(jt)).radius] = cart2pol(streamObjList{k}.channelSpace(jt,1), streamObjList{k}.channelSpace(jt,2), ...
                                    streamObjList{k}.channelSpace(jt,3));
                                chanlocs(locChannels(jt)).theta = -chanlocs(locChannels(jt)).theta*180/pi;
                            end
                        end
                        channelType = 'EEG';
                    case 'dataStream', channelType = 'EEG';
                    case 'mocap',      channelType = 'Mocap';
                    case 'wii',        channelType = 'Wii';
                    otherwise,         channelType = class(streamObjList{k});
                end
                for jt=1:streamObjList{k}.numberOfChannels, chanlocs(locChannels(jt)).type = channelType;end
            end
            EEG.chanlocs = chanlocs;
            EEG.etc.mobi.sessionUUID = streamObjList{1}.sessionUUID;
            
            if ~isempty(latency)
                [latency,loc] = sort(latency,'ascend');
                type = type(loc);
                hedTag = hedTag(loc);
                Nevents = length(latency);
                disp(['Inserting ' num2str(Nevents) ' events.']);
                obj.container.initStatusbar(1,Nevents,'Creating EEG.event...');
                EEG.event = repmat(struct('type','','latency',0,'duration',0,'urevent',1,'hedTag',[]),1,Nevents);
                for it=1:Nevents
                    EEG.event(it).type = type{it};
                    EEG.event(it).latency = latency(it);
                    EEG.event(it).hedTag = hedTag{it};
                    EEG.event(it).urevent = it;
                    if strcmp(type,'boundary'), EEG.event(it).duration = NaN;end
                    obj.container.statusbar(it);
                end
            end
            EEG.urevent = EEG.event;
            if ~isempty(newEEGfile)
                pop_saveset( EEG, [name '.set'],path);
                EEG = pop_loadset( [name '.set'],path);
            end
            if nargout < 1 && updateGui
                try 
                    ALLEEG = evalin('base','ALLEEG');
                catch
                    ALLEEG = [];
                end
                [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);
                assignin('base','ALLEEG',ALLEEG);
                assignin('base','CURRENTSET',CURRENTSET);
                assignin('base','EEG',EEG);
                try
                    evalin('base','eeglab(''redraw'');');
                end
            end
        end
    end
    %%
    methods(Static)
        function updateTree(~,evnt)
            evnt.AffectedObject.item{end}.container = evnt.AffectedObject;
            evnt.AffectedObject.updateLogicalStructure;
            evnt.AffectedObject.gObj = [];
            if evnt.AffectedObject.container.isGuiActive
                [~,evnt.AffectedObject.gObj] = viewLogicalStructure(evnt.AffectedObject,'',false);
            end
            evnt.AffectedObject.save;
        end
        function checkThisFolder(folder)
            if ~exist(folder,'dir')
                try mkdir(folder);
                catch ME
                    ME.rethrow;
                end
            end
            if numel(dir(folder))
                files = dir(folder);
                files(1:2) = [];
                files = {files.name};
                if isempty(files), return;end
                warning('MoBILAB:notEmptyFolder','MoBILAB needs an empty folder to start a new session. All the existent files in this folder will be compressed.');
                zipfile = [folder filesep 'lost+found.zip'];
                disp(['Zipping: ' zipfile '...']);
                if exist(zipfile,'file'), zipfile = [folder filesep 'lost+found2.zip'];end
                zip(zipfile,files,folder);
                for it=1:length(files), java.io.File([folder filesep files{it}]).delete();end
                java.io.File(zipfile).renameTo(java.io.File([folder filesep 'lost+found.zip']));
            end
        end
    end
    %%
    methods(Hidden = true)
        %%
        function initStatusbar(obj,mn,mx,msg)
            if nargin < 4, error('Not enough input arguments.');end
            try obj.container.initStatusbar(mn,mx,msg);end %#ok
        end
        function statusbar(obj,val,msg)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3
                 obj.container.statusbar(val);
            else obj.container.statusbar(val,msg);
            end
        end
        function val     = isGuiActive(obj), val     = obj.container.isGuiActive;end
        function hFigure = refreshGui(obj),  hFigure = obj.container.gui;end
        %%
        function indices = getIndices4aBranch(obj,index)
            indices = obj.gObj.getIndex4aBranch(index+1)-1;
            indices(indices==0) = [];
        end
        %%
        function obj = connect(obj)
            if isempty(obj.item), return;end
            N = length(obj.item);
            try for it=1:N, obj.item{it}.connect;end
            catch ME, ME.rethrow;
            end
        end
        function obj = disconnect(obj)
            N = length(obj.item);
            try for it=1:N, obj.item{it}.disconnect;end
            catch ME, ME.rethrow;
            end
        end
        %%
        function dim = size(obj), dim = size(obj.item);end
        function mem = memory(obj), A = dir(obj.mobiDataDirectory);mem = sum(cell2mat({A.bytes}));end %/10^9;
        %%
        function string = disp(obj)
            if ~isvalid(obj), disp('Invalid or deleted object.');return;end
            string = sprintf('Class: %s\n\nProperties:\n  mobiDataDirectory:  %s\n  sessionUUID: %s\n',...
                class(obj),obj.mobiDataDirectory,obj.sessionUUID);
            disp(string)
            N = length(obj.item);
            cmd = '';
            if N > 8, N=8;cmd='fprintf(''\n.\n.\n.\n and many, many more. Use mobilab.gui to see all of them.\n'')';end
            for it=1:N
                fprintf('\nitem{%i}:\n',it);
                obj.item{it}.disp;
            end
            eval(cmd);
        end
        %%
        function linkData(obj)
            N = length(obj.item);
            if N < 1, return;end
            for it=1:N
                obj.item{it}.sessionUUID = obj.sessionUUID;
                saveProperty(obj.item{it},'sessionUUID',obj.sessionUUID)
            end
        end
        %%
        function updateLogicalStructure(obj)
            N = length(obj.item);
            obj.hashTable = cell(N,1);
            obj.logicalStructure = zeros(N+1);
            for it=1:N, obj.hashTable{it} = char(obj.item{it}.uuid);end
            for it=1:N
                if isempty(obj.item{it}.parent), index = 1;
                else index = obj.findItem(obj.item{it}.parent.uuid)+1;
                end
                obj.logicalStructure(it+1,index) = 1;
                obj.logicalStructure(index,it+1) = 1;
            end
        end
        %%
        function [figureHandle,gObj] = viewLogicalStructure(obj,callback,showTreeFlag)
            if nargin < 2, callback = 'dispNode_Callback';end
            if nargin < 3, showTreeFlag = true;end
            gObj = obj.gObj;
            if showTreeFlag
                figureHandle = obj.container.gui(callback);
            else figureHandle = [];
            end
        end
        %%
        function [index,uuid] = getItemIndexFromSourceId(obj,sourceId)
            index = 0;
            uuid = '';
            N = length(obj.item);
            if ~N, return;end
            if ~isfield(obj.item{1}.hardwareMetaData,'source_id'), return;end
            
            sourceId_all = cell(N,1);
            for it=1:N, sourceId_all{it} = obj.item{it}.hardwareMetaData.source_id;end
            I = cellfun(@ischar,sourceId_all);
            sourceId_all(~I) = repmat({'-1-1-1'},sum(~I),1);
            index = find(ismember(sourceId_all,sourceId));
            if nargout > 1
                uuid = cell(length(index),1);
                for it=1:length(index), uuid{it} = obj.item{index(it)}.uuid;end
                if length(uuid) == 1, uuid = uuid{1};end
            end
        end
        %%
        function [index,uuid] = getItemIndexFromHostname(obj,hostname)
            index = 0;
            uuid = '';
            N = length(obj.item);
            if ~N, return;end
            if ~isfield(obj.item{1}.hardwareMetaData,'hostname'), return;end
            hostname_all = cell(N,1);
            for it=1:N, hostname_all{it} = obj.item{it}.hardwareMetaData.hostname;end
            I = cellfun(@ischar,hostname_all);
            hostname_all(~I) = repmat({'-1-1-1'},sum(~I),1);
            index = find(ismember(hostname_all,hostname));
            if nargout > 1
                uuid = cell(length(index),1);
                for it=1:length(index), uuid{it} = obj.item{index(it)}.uuid;end
                if length(uuid) == 1, uuid = uuid{1};end
            end
        end
        %%
        function cobj = makeMultiMarkerStreamObject(obj,delList)
            if nargin < 2, delList = false;end
            markerIndices = obj.getItemIndexFromItemClass('markerStream');
            if isempty(markerIndices), return;end
            
            markerStreamList = obj.item(markerIndices);
            N = length(markerStreamList);
            labels = {};
            timeStamp = [];
            for it=1:N
                timeStamp = [timeStamp markerStreamList{it}.timeStamp]; %#ok
                tmp = markerStreamList{it}.event.uniqueLabel(:);
                for jt=1:length(tmp), tmp{jt} = [markerStreamList{it}.name '/' tmp{jt}];end
                labels = cat(1,labels,tmp);
            end
            timeStamp = unique(timeStamp);
            timeStamp = unique([timeStamp-0.002 timeStamp timeStamp+0.002]);
            labels = sort(labels);
            
            path = obj.mobiDataDirectory;
            metadata = markerStreamList{it}.saveobj;
            uuid = generateUUID;
            metadata.uuid = uuid;
            metadata.name = 'multiMarker';
            metadata.writable = false;
            metadata.label = labels;
            metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
            metadata.timeStamp = timeStamp;
            metadata.numberOfChannels = length(labels);
            metadata.event = event;
            metadata.class = 'multiMarkerStream';
            metadata.writable = true;
            data = zeros(length(metadata.timeStamp),metadata.numberOfChannels);
            fid = fopen(metadata.binFile,'w');
            fwrite(fid,data(:),markerStreamList{it}.precision);
            fclose(fid);
            header = metadata2headerFile(metadata);
            cobj = obj.addItem(header);
            
            for it=1:N
                uLabel = markerStreamList{it}.event.uniqueLabel;
                for jt=1:length(uLabel)
                    latency = markerStreamList{it}.event.getLatencyForEventLabel(uLabel{jt});
                    latency = markerStreamList{it}.timeStamp(latency);
                    indices = cobj.getTimeIndex(latency);
                    channel = strcmp(cobj.label,[markerStreamList{it}.name '/' uLabel{jt}]);
                    cobj.mmfObj.Data.x(indices,channel) = 1;
                end
            end
            
            eventObj = event;
            obj.container.initStatusbar(1,N,'Concatenating hedTags...');
            for it=1:N
                hedTag         = markerStreamList{it}.event.hedTag;
                latencyInFrame = markerStreamList{it}.event.latencyInFrame;
                for jt=1:length(hedTag), hedTag{jt} = [markerStreamList{it}.name '/' hedTag{jt}];end
                latency  = markerStreamList{it}.timeStamp(latencyInFrame);
                indices  = cobj.getTimeIndex(latency);
                eventObj = eventObj.addEvent(indices,hedTag);
                obj.container.statusbar(it);
            end
            cobj.event = eventObj;
            if delList, obj.deleteItem(markerIndices);end
        end
        
        %%
        function findSpaceBoundary(obj)
            N = length(obj.item);
            if N < 1, return;end
            roomSize = [];
            for it=1:N
                if isa(obj.item{it},'mocap')
                    if isempty(roomSize)
                        I = 1:10:size(obj.item{it},1);
                        
                        mx = max(max(abs(squeeze(obj.item{it}.dataInXYZ(I,1,:)))));
                        roomSize(1,:) = [-mx mx];
                        
                        mx = max(max(abs(squeeze(obj.item{it}.dataInXYZ(I,2,:)))));
                        roomSize(2,:) = [-mx mx];
                        
                        mx = max(max(squeeze(obj.item{it}.dataInXYZ(I,3,:))));
                        % mn = min(min(squeeze(obj.item{it}.dataInXYZ(I,3,:))));
                        roomSize(3,:) = [0 mx];
                        
                        roomSize = roomSize*1.1;
                        obj.item{it}.animationParameters.limits = roomSize;
                    else
                        obj.item{it}.animationParameters.limits = roomSize;
                    end
                end
            end
        end
    end
end


%% -------------------------------------------------------------------------
function [timeStamps, Ntimepoints,Nchannels,labels,streamObj,type,latency,hedTag] = alignStreams(EEG,streamObj,eventcodesObj)
% bytes = dir(streamObj.binFile);
% bytes = bytes.bytes/1e9;
% if bytes < 0.5
fid = fopen(EEG.data,'w');
if fid < 0, return;end
try
    if ~iscell(streamObj), streamObj = {streamObj};end
    
    N = length(streamObj);
    limits = zeros(N,2);
    totalNumberOfChannels = 0;
    labels = [];
    names = [];
    for it=1:N
        limits(it,:) = streamObj{it}.timeStamp([1 end]);
        totalNumberOfChannels = totalNumberOfChannels+streamObj{it}.numberOfChannels;
        labels = cat(1,labels,streamObj{it}.label(:));
        tmp = repmat({streamObj{it}.name},streamObj{it}.numberOfChannels,1);
        names = cat(1,names,tmp);
    end
    
    if totalNumberOfChannels > length(unique(labels))
        for it=1:totalNumberOfChannels, labels{it} = [names{it} '_' labels{it}];end %#ok
    end
    
    limits = [max(limits(:,1)) min(limits(:,2))];
    [t0,tn] = streamObj{1}.getTimeIndex(limits);
    xi = streamObj{1}.timeStamp(t0:tn)';
    timeStamps = xi-xi(1);
    disc = abs(1./diff(xi)/streamObj{1}.samplingRate-1);
    if any(disc > 0.24)
        warning('There may be discontinuities in the data, check EEG.times to be sure and add boundary events if that is the case.');
    end
    Nxi = length(xi);
    precision = 'single';
    
    for it=1:N, if ~streamObj{it}.isMemoryMappingActive, fclose(fid); error('The stream is empty.');end;end
    
    if N > 1, streamObj{1}.container.container.initStatusbar(1,N,'Aligning streams...');end
    [~,filename] = fileparts(tempname);
    tmpFile = [fileparts(EEG.data) filesep filename];
    tfid = fopen(tmpFile,'w');
    for it=1:N        
        y = streamObj{it}.mmfObj.Data.x;  % saving memory, streamObj{it}.data will create in memory the variable, streamObj{it}.mmfObj.Data.x is a lazy copy
        ind = unique(streamObj{it}.getTimeIndex(xi));
        x = streamObj{it}.timeStamp(ind)';
        for ch=1:streamObj{it}.numberOfChannels
            yi = interp1(x,y(ind,ch),xi,'linear');
            fwrite(tfid,yi(:),precision);
        end
        % if srOld, streamObj{it}.samplingRate = srOld;end
        if N > 1, streamObj{1}.container.container.statusbar(it);end
    end
    fclose(tfid);
    mmfObj = memmapfile(tmpFile,'Format',{precision [Nxi totalNumberOfChannels] 'x'},'Writable',false);
    data = mmfObj.Data.x;
    
    bufferSize = 2048;
    streamObj{1}.container.container.initStatusbar(1,Nxi,'Creating EEG.data...');
    for it=1:bufferSize:Nxi
        if it+bufferSize-1 <= Nxi, writeThis = data(it:it+bufferSize-1,:)';
        else writeThis = data(it:end,:)';
        end
        fwrite(fid,writeThis(:),precision);
        streamObj{1}.container.container.statusbar(it);
    end
    streamObj{1}.container.container.statusbar(inf);
    fclose(fid);
    clear mmfObj
    java.io.File(tmpFile).delete();
    Ntimepoints = Nxi;
    Nchannels = totalNumberOfChannels;
    
    type = streamObj{1}.event.label(:);
    latency = streamObj{1}.timeStamp(streamObj{1}.event.latencyInFrame(:));
    hedTag = streamObj{1}.event.hedTag(:);
    
    for it=2:length(streamObj)
        type = cat(1,type,streamObj{it}.event.label(:));
        hedTag = cat(1,hedTag,streamObj{it}.event.hedTag(:));
        latency = [latency streamObj{1}.timeStamp(streamObj{it}.event.latencyInFrame(:))]; %#ok
    end
    
    for it=1:length(eventcodesObj)
        if ~isempty(eventcodesObj{it}.event.latencyInFrame)
            type = cat(1,type,eventcodesObj{it}.event.label(:));
            hedTag = cat(1,hedTag,eventcodesObj{it}.event.hedTag(:));
            tmp = eventcodesObj{it}.timeStamp(eventcodesObj{it}.event.latencyInFrame);
            tmp = streamObj{1}.getTimeIndex(tmp);
            latency = [latency streamObj{1}.timeStamp(tmp)]; %#ok
        end
    end
    
    I = latency < limits(1) | latency > limits(2);
    type(I) = [];
    latency(I) = [];
    hedTag(I) = [];
    if ~isempty(latency), latency = streamObj{1}.getTimeIndex(latency)-(t0-1);else return;end
    
    list = [latency(:) latency(:) latency(:)];
    uType = unique(type);
    for it=1:length(uType)
        I = ismember(type,uType{it});
        list(I,2) = it;
    end
    uHed = unique(hedTag);
    for it=1:length(uHed)
        I = ismember(hedTag,uHed{it});
        list(I,3) = it;
    end
    
    [~,I] = unique(list,'rows');
    type = type(I);
    latency = latency(I);
    hedTag = hedTag(I);
catch ME
    ME.rethrow;
    if exist(EEG.data,'file'), delete(EEG.data);end
    if exist('tmpFile','var') && exist(tmpFile,'file'), delete(tmpFile);end
end
end