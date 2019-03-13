% Definition of the class dataSourceMoBI. This class reads into MoBILAB the
% content of a folder containing MoBILAB files.
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 05-Apr-2011

%%
classdef dataSourceMoBI < dataSource
    methods
        %%
        function obj = dataSourceMoBI(mobiDataDirectory)
            % Creates a dataSourceMoBI object.
            % 
            % Input arguments:
            %       folder: folder containing MoBILAB files 
            % 
            % Output arguments:
            %       obj: dataSource object (handle)
            %
            % Usage:
            %        % folder: folder containing MoBILAB files 
            %        obj = dataSourceMoBI( folder );

            if nargin < 1, error('No enough input arguments.');end
            mobiDataDirectory = strtrim(mobiDataDirectory);    
            folder = dir(mobiDataDirectory);
            files = {folder.name};
            I = strfind(files,'.hdr');
            I = cellfun(@isempty,I);
            files(I) = [];
            if isempty(files), error('The folder is empty.');end
            loc = find(files{1} == '-');
            if length(loc) < 8
                try 
                    warning off %#ok
                    load([mobiDataDirectory filesep files{1}],'-mat','sessionUUID');
                    warning on; %#ok
                    if ~exist('sessionUUID','var'), error('old format');end 
                catch %#ok
                    metadata = load([mobiDataDirectory filesep files{1}],'-mat');
                    if isfield(metadata,'sessionUUID')
                        sessionUUID = metadata.sessionUUID;
                    else
                        uuid = generateUUID;
                        sessionUUID = uuid;
                    end
                end
                for it=1:length(files)
                    oldHeader  = fullfile(mobiDataDirectory,[files{it}(1:end-4) '.hdr']);
                    oldBinFile = fullfile(mobiDataDirectory,[files{it}(1:end-4) '.bin']);
                    newHeader  = fullfile(mobiDataDirectory,[files{it}(1:end-4) '_' sessionUUID '.hdr']);
                    newBinFile = fullfile(mobiDataDirectory,[files{it}(1:end-4) '_' sessionUUID '.bin']);
                    java.io.File(oldHeader ).renameTo(java.io.File(newHeader));
                    java.io.File(oldBinFile).renameTo(java.io.File(newBinFile));
                end
            else loc = find(files{1} == '_');
                sessionUUID = files{1}(loc(end)+1:end-4);
            end
            if exist([mobiDataDirectory filesep 'descriptor.MoBI'],'file')
                load([mobiDataDirectory filesep 'descriptor.MoBI'],'-mat');
                try txt = char(datasource.notes); %#ok
                    save([mobiDataDirectory filesep 'notes_' sessionUUID '.txt'],'-ascii','txt');
                end
            end
            
            %% constructing the dataSource object
            obj@dataSource(mobiDataDirectory,sessionUUID);
            obj.listenerHandle.Enabled = false;
               
            headers = pickfiles(mobiDataDirectory,{sessionUUID '.hdr'});
            N = size(headers,1);
            obj.container.initStatusbar(1,N,'Loading files...');
            for it=1:N
                header = deblank(headers(it,:));
                if obj.checkHeader(header), obj.addItem(header);end
                obj.container.statusbar(it);
            end
            dob = zeros(1,length(obj.item));
            for it=1:length(obj.item), dob(it) = obj.item{it}.dob;end
            [~,loc] = sort(dob);
            obj.item = obj.item(loc); 
            
            obj.connect;
            obj.updateLogicalStructure;
            
            [~,obj.gObj] = viewLogicalStructure(obj,'',false);
            if isempty(obj.item{1}.sessionUUID), obj.linkData;end
            obj.listenerHandle.Enabled = true;
        end
        %%
        function val = checkHeader(obj,header)
            if isempty(header), error('This folder is corrupted. Cannot find the headers.');end
            val = true;
            version = coreStreamObject.version;
            try
                warning off 
                load(header,'-mat','hdrVersion');
                warning on  
            catch hdrVersion = -inf; %#ok
            end
            
            if ~exist('hdrVersion','var'), hdrVersion = -inf;end
            if version == hdrVersion, return;end
            hdrVersion = version; %#ok
            save(header,'-mat','-append','hdrVersion');
            
            metadata = load(header,'-mat');
            if isfield(metadata,'metadata'), metadata = metadata.metadata;end
            if ~exist('metadata','var')
                val = false;
                binFile = [header(1:end-3) 'bin'];
                if exist(binFile,'file')
                     toZip = {header,binFile};
                else
                    toZip = header;
                end
                zipfile = [obj.mobiDataDirectory filesep 'lost+found.zip'];
                fprintf(['File ' header ' may be corrupt.\nIt will be added to ' zipfile '\n']);
                if exist(zipfile,'file'), zipfile = [obj.mobiDataDirectory filesep 'lost+found2.zip'];end
                zip(zipfile,toZip,obj.mobiDataDirectory);
                java.io.File(zipfile).renameTo(java.io.File([obj.mobiDataDirectory filesep 'lost+found.zip']));
                delete(header);
                if exist(binfile,'file'), delete(binFile);end
                return;
            end
            saveThis = false;
            if isfield(metadata,'eventInStruct')
                metadata.event = metadata.eventInStruct;
                metadata = rmfield(metadata,'eventInStruct');
                saveThis = true;
            end
            if ~isfield(metadata.event,'hedTag') 
                metadata.event.hedTag = metadata.event.label;
                saveThis = true;
            end
            if ~isfield(metadata,'class')
                if ~isempty(strfind(metadata.name,'biosemi')) || ~isempty(strfind(metadata.name,'eeg'))
                    metadata.class = 'eeg';
                elseif ~isempty(strfind(metadata.name,'phasespace')) || ~isempty(strfind(metadata.name,'mocap'))
                    metadata.class = 'mocap';
                else metadata.class = 'dataStream';
                end
                saveThis = true;
            end
            if strcmp(metadata.class,'dataStream') && (~isempty(strfind(metadata.name,'biosemi')) || ~isempty(strfind(metadata.name,'eeg')))
                metadata.class = 'eeg';
                saveThis = true;
            end
            if isfield(metadata,'hardwareMetaDataObj') 
                metadata.hardwareMetaData = metadata.hardwareMetaDataObj;
                metadata = rmfield(metadata,'hardwareMetaDataObj');
                saveThis = true;
            end
            if ~isfield(metadata,'parentCommand')
                metadata.parentCommand = [];
                saveThis = true;
            end
            if ~isempty(metadata.parentCommand) && isfield(metadata.parentCommand,'uuid') && ~ischar(metadata.parentCommand.uuid)
                metadata.parentCommand.uuid = char(metadata.parentCommand.uuid);
                saveThis = true;
            end
            if strcmp(metadata.class,'vectorMeasure') || strcmp(metadata.class,'segmentedMocap')
                metadata.class = 'mocap';
                saveThis = true;
            end
            if ~isfield(metadata,'binFile')
                if ~isfield(metadata,'mmfName'), metadata.mmfName = [header(1:end-4) '.bin'];end
                metadata.binFile = metadata.mmfName;
                metadata = rmfield(metadata,'mmfName');
                saveThis = true;
            end
            if ~strcmp(metadata.binFile(1:end-4),header(1:end-4))
                metadata.binFile = [header(1:end-4) '.bin'];
                metadata.header = header;
                saveThis = true;
            end
            if ~isfield(metadata,'sessionUUID')
                if ~isempty(obj.item)
                    metadata.sessionUUID = obj.item{1}.sessionUUID;
                else
                    uuid = generateUUID;
                    metadata.sessionUUID = uuid;
                end
                saveThis = true;
            end
            if ~ischar(metadata.uuid), metadata.uuid = char(metadata.uuid);saveThis = true;end
            if ~ischar(metadata.sessionUUID), metadata.sessionUUID = char(metadata.sessionUUID);saveThis = true;end
            if isfield(metadata,'segmentUUID') && ~ischar(metadata.segmentUUID), metadata.segmentUUID = char(metadata.segmentUUID);saveThis = true;end
            if isfield(metadata,'originalStreamObj') && ~ischar(metadata.originalStreamObj), metadata.originalStreamObj = char(metadata.originalStreamObj);saveThis = true;end
            
            binFile = pickfiles(obj.mobiDataDirectory,{metadata.uuid obj.sessionUUID '.bin'});
            if ~strcmp(binFile,metadata.binFile)
                metadata.binFile = binFile;
                metadata.header  = header;
                saveThis = true;
            end
            if isfield(metadata,'segmentObj')
                id = obj.segmentList.getSegmentID(metadata.segmentObj);
                if isempty(id), metadata.segmentUUID = obj.segmentList.addSegment(metadata.segmentObj);
                else metadata.segmentUUID = id;
                end
                metadata = rmfield(metadata,'segmentObj');
                saveThis = true;
            end
            if saveThis
                disp(['Updating: ' header]);
                metadata2headerFile(metadata);
            end
            clear metadata
        end
    end
end


%%---------------------------------------
function outFile = look4it(file,name)
outFile = [];
[path,~,ext] = fileparts(file);

tmpFiles = pickfiles(path,{name ext});
if iscell(tmpFiles)
    for it=1:length(tmpFiles)
        [~,filename] = fileparts(tmpFiles{it});
        indTmp = strfind(filename,name);
        if indTmp==1
            outFile = tmpFiles{it};
            break
        end
    end
elseif ischar(tmpFiles) && size(tmpFiles,1)==1
    [~,filename] = fileparts(tmpFiles);
    indTmp = strfind(filename,name);
    if indTmp==1
        outFile = tmpFiles;
    end
else
    for it=1:size(tmpFiles,1)
        [~,filename] = fileparts(deblank(tmpFiles(it,:)));
        indTmp = strfind(filename,name);
        if indTmp==1
            outFile = deblank(tmpFiles(it,:));
            break
        end
    end
end
end
