% Definition of the class dataSourceDRF. This class imports into MoBILAB files
% recorder by Datariver saved in drf format.
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 05-Apr-2011

%%
classdef dataSourceDRF < dataSource
    methods
        %%
        function obj = dataSourceDRF(varargin)
            % Creates a dataSourceDRF object.
            % 
            % Input arguments:
            %       file:              drf file recorded by Datariver 
            %       mobiDataDirectory: path to the directory where the collection
            %                          of  files will be stored
            %                            
            % Output arguments:         
            %       obj:               dataSource object (handle)
            %
            % Usage:
            %        % file: drf file recorded by Datariver
            %        % mobiDataDirectory: path to the directory where the collection of
            %        %                    files are or will be stored.
            %
            %        obj = dataSourceDRF( file,  mobiDataDirectory);
            
            if nargin==1, varargin = varargin{1};end
            if length(varargin) < 2, error('Not enough input arguments.');end
            sourceFileName = varargin{1};
            mobiDataDirectory = varargin{2};
            if length(varargin) == 3
                tmpDir = varargin{3};
                if ~exist(tmpDir,'dir') && ~isempty(tmpDir), mkdir(tmpDir);end
            else tmpDir = '';
            end
            
            [drfFolder,drfName,ext] = fileparts(sourceFileName);
            if ~strcmp(ext,'.drf'), error('prog:input',['dataSourceDRF cannot read ''' ext ''' format.']);end
            if ~exist([drfFolder filesep drfName '.xml'],'file'), error('prog:input','Unable to read the correspondent xml header.');end
            if strcmp(drfFolder,mobiDataDirectory), error('Source and destiny folders must be different.');end
            
            uuid = generateUUID;
            obj@dataSource(mobiDataDirectory,uuid);
            obj.checkThisFolder(mobiDataDirectory);
            
            % Some internal parameters
            buffer_size    = 4*2*2048; % only 16 Mbyte at a time in RAM  (2*2048 int32 elements)
            buffer_counter = 1;
            stream_count   = 1;
            streamsList = {'biosemi' 'phasespace' 'videostream' 'audiosend' 'wii' 'eventcodes' 'hotgazestream' 'scenestream'};
            parentCommand.commandName = 'dataSourceDRF';
            parentCommand.varargin{1} = sourceFileName;
            parentCommand.varargin{2} = mobiDataDirectory;
            eegCount = 0;
            mocapCount = 0;
            metadata.sessionUUID = obj.sessionUUID;
                       
            % Preparing for the show!!!
            xDom = xmlread([drfFolder filesep drfName '.xml']);
            fid = fopen(sourceFileName,'r');
            cleanupHandle = onCleanup(@()fclose(fid));
            if fid<0, error('prog:input',['Could not read from: ' sourceFileName]);end
            
            % Parsing the xml header
            allfilesamples = xDom.getElementsByTagName('filesample');
            reservedItem = xDom.getElementsByTagName('reserved');
            for it=0:allfilesamples.getLength-1
                tmp = allfilesamples.item(it).getElementsByTagName('stream_count');
                offset = str2double(tmp.item(0).getAttribute('bytes'));
                fseek(fid,offset,'cof');
                
                allsamples = allfilesamples.item(it).getElementsByTagName('sample');
                for jt=0:allsamples.getLength-1
                    allstreams = allsamples.item(jt).getElementsByTagName('stream');
                    for k=0:allstreams.getLength-1
                        convert_to_uV{stream_count} = [];%#ok
                        labels{stream_count}        = [];%#ok
                        name{stream_count}          = lower(char(allstreams.item(k).getAttribute('name'))); %#ok %data_type
                        tmpInd = find(name{stream_count} == '\');
                        if ~isempty(tmpInd), name{stream_count} = name{stream_count}(tmpInd(end)+1:end);end %#ok
                        samplingRate = str2double(allstreams.item(k).getAttribute('sampling_rate'));
                        data_tag      = allstreams.item(k).getElementsByTagName('data');
                        data_child    = data_tag.item(0).getChildNodes;
                        data_items    = str2double(data_tag.item(0).getAttribute('items'));
                        item_size_tag = allstreams.item(k).getElementsByTagName('item_size');
                        hardwareMetaDataObj = hardwareMetaData;
                        hardwareMetaDataObj.reserved_bytes = char(reservedItem.item(0).getAttribute('bytes'));
                        hardwareMetaDataObj.offset = offset;
                        hardwareMetaDataObj.data_items = data_items;
                        hardwareMetaDataObj.item_size = str2double(item_size_tag.item(0).getAttribute('value'));
                        hardwareMetaDataObj.sampling_rate = samplingRate;
                        hardwareMetaDataObj.bytesHeader = str2double(allstreams.item(k).getAttribute('bytes'));
                        add_label    = 1;
                        for kk=0:data_child.getLength-1
                            try%#ok
                                rep_lb = str2double(data_child.item(kk).getAttribute('count'));
                                uV = 1./str2double(data_child.item(kk).getAttribute('uV'));
                                if isempty(uV) || isnan(uV), uV = 1;end
                                if isempty(rep_lb), hardwareMetaDataObj.count(end+1) = 1;
                                else hardwareMetaDataObj.count(end+1) = rep_lb;
                                end
                                hardwareMetaDataObj.uV(end+1) = 1/uV;
                                tmplabels = [];
                                for kkk=0:data_child.item(kk).getLength-1
                                    try %#ok
                                        rd_label = char(data_child.item(kk).item(kkk).getAttribute('label'));
                                        if ~isempty(rd_label), rd_label = strtrim(rd_label);end
                                        hardwareMetaDataObj.label{end+1} = rd_label;
                                        ind = strfind(rd_label,'%');
                                        if ~isempty(ind), rd_label = rd_label(1:ind-1);end
                                        tmplabels = cat(1,tmplabels,{rd_label});
                                        add_label = add_label + 1;
                                    end
                                end
                                repLabels = [];
                                if isempty(rep_lb)
                                    repLabels = char(data_child.item(kk).getNodeName);
                                    atrib = data_child.item(kk).getAttributes;
                                    for atrib_iter=0:atrib.getLength-1, if ~isempty(atrib.item(atrib_iter).toString), repLabels = [repLabels '_' char(atrib.item(atrib_iter).toString)];end;end %#ok
                                    if ~isempty(repLabels), repLabels = {repLabels};end
                                else for k4=1:rep_lb, for k5=1:length(tmplabels), repLabels{end+1,1} = [tmplabels{k5} num2str(k4)];end;end %#ok
                                end
                                labels{stream_count} = cat(1,labels{stream_count},repLabels); %#ok
                                convert_to_uV{stream_count} = cat(1,convert_to_uV{stream_count},repmat(uV,length(repLabels),1));%#ok
                            end
                        end
                        
                        % creating the dataSource.item with the first samples
                        [samples,timeStamp,event,numberOfChannels] = read_stream(fid);
                        if numberOfChannels > 1
                            if numberOfChannels ~= str2double(data_child.getAttribute('items'))  % length(labels{stream_count})
                                error('Cannot read the file, there is a mismatch between the header and the content of the drf file.');
                            end
                        end
                        if isempty(convert_to_uV{stream_count}),
                            if numberOfChannels > 0, convert_to_uV{stream_count} = ones(numberOfChannels,1);%#ok
                            else convert_to_uV{stream_count} = 1;%#ok
                            end
                        end
                        
                        binFile = [tempname '.bin'];
                        if ~isempty(tmpDir)
                            [~,tmpName] = fileparts(binFile);
                            binFile = fullfile(tmpDir,tmpName);
                        end
                        if ~exist(binFile,'file'), fid2 = fopen(binFile,'w');fclose(fid2);end
                        [filepath,filename] = fileparts(binFile);
                        header = fullfile(filepath,[filename '.hdr']);
                        streamClass = '';
                        for sList=1:length(streamsList)
                            if ~isempty(strfind(lower(name{stream_count}),streamsList{sList})), streamClass = streamsList{sList};end
                        end
                        hardwareMetaDataObj.name = streamClass;
                        metadata.binFile = binFile;
                        metadata.header = header;
                        metadata.dob = now;
                        metadata.name = name{stream_count};
                        metadata.timeStamp = [];
                        if stream_count==1, metadata.samplingRate = samplingRate;end
                        metadata.numberOfChannels = numberOfChannels;
                        metadata.label = labels{stream_count};
                        metadata.writable = false;
                        metadata.parentCommand = parentCommand;
                        metadata.hardwareMetaData = hardwareMetaData;
                        metadata.precision = 'double';
                        metadata.event.label = cell(0,1);
                        metadata.event.hedTag = cell(0,1);
                        metadata.event.latencyInFrame = [];
                        uuid = generateUUID;
                        metadata.uuid = uuid;
                        metadata.unit = 'none';
                        switch streamClass
                            case 'biosemi'
                                if eegCount==0, eegName = 'eeg'; else eegName = ['eeg' num2str(eegCount)];end
                                metadata.name = eegName;
                                metadata.unit = 'microvolts';
                                metadata.class = 'eeg';
                                metadata.auxChannel.label = {};
                                metadata.auxChannel.data = [];
                                header = metadata2headerFile(metadata);
                                metadata = rmfield(metadata,'auxChannel');
                                eegCount = eegCount+1;
                                
                            case 'phasespace'
                                if mocapCount==0, mocapName = 'mocap'; else mocapName = ['mocap' num2str(mocapCount)];end
                                metadata.name = mocapName;
                                metadata.animationParameters = struct('limits',[],'conn',[],'bodymodel',[]);
                                metadata.class = 'mocap';
                                header = metadata2headerFile(metadata);
                                metadata = rmfield(metadata,'animationParameters');
                                mocapCount = mocapCount+1;
                                
                            case 'wii'
                                metadata.animationParameters = struct('limits',[],'conn',[],'bodymodel',[]);
                                metadata.class = 'wii';
                                header = metadata2headerFile(metadata);
                                metadata = rmfield(metadata,'animationParameters');
                                
                            case 'videostream'
                                metadata.numberOfChannels = 1;
                                metadata.label = {'video frame'};
                                metadata.class = 'videoStream';
                                header = metadata2headerFile(metadata);
                                
                            case 'scenestream'
                                metadata.numberOfChannels = 1;
                                metadata.label = {'video frame'};
                                metadata.class = 'videoStream';
                                header = metadata2headerFile(metadata);
                                
                            case 'hotgazestream'
                                metadata.class = 'gazeStream';
                                header = metadata2headerFile(metadata);
                                
                            case 'eventcodes'
                                metadata.class = 'markerStream';
                                header = metadata2headerFile(metadata);

                            otherwise
                                metadata.class = 'dataStream';
                                header = metadata2headerFile(metadata);

                        end
                        obj.addItem(header);
                        obj.item{stream_count}.addSamples(convert_to_uV{stream_count}.*samples,timeStamp,event);
                        stream_count = stream_count + 1;
                    end
                end
            end
            
            % Reading from the binary file until the end
            N = stream_count-1;
            samples   = cell(N,1);
            timeStamp = cell(N,1);
            event     = cell(N,1);
            oneTimeWarning = 1;
            
            obj.container.lockGui('Reading...');
            while ~feof(fid)
                try fseek(fid,offset,'cof');
                    for stream_count=1:N
                        [tmpSamples,timeStamp{stream_count}(buffer_counter),event{stream_count}(buffer_counter)] = read_stream(fid);
                        try samples{stream_count}(:,buffer_counter) = tmpSamples;
                        catch ME
                            if strcmp(ME.identifier,'MATLAB:subsassigndimmismatch') && obj.item{stream_count}.numberOfChannels > 0
                                samples{stream_count}(:,buffer_counter) = nan;
                                if oneTimeWarning
                                    warning('MoBILAB:corruptedFile',['From sample ' num2str(length(obj.item{stream_count}.timeStamp))...
                                        ' there are no more data in stream ' obj.item{stream_count}.name '.']);
                                    oneTimeWarning = 0;
                                end
                            elseif obj.item{stream_count}.numberOfChannels ~= 0
                                error('MoBILAB:corruptedFile','The file is corrupted.');
                            end
                        end
                    end
                    buffer_counter = buffer_counter+1;
                    if buffer_counter > buffer_size
                        for stream_count=1:N
                            if ~isempty(samples{stream_count})
                                obj.item{stream_count}.addSamples(repmat(convert_to_uV{stream_count},...
                                    buffer_size,1).*samples{stream_count}(:),timeStamp{stream_count}(:),...
                                    event{stream_count}(:));
                            else obj.item{stream_count}.addSamples([],timeStamp{stream_count}(:),event{stream_count}(:));
                            end
                        end
                        buffer_counter = 1;
                    end
                catch ME
                    if  strcmpi(ME.identifier,'MoBILAB:corruptedFile') || strcmpi(ME.identifier,'MATLAB:badprecision_mx')
                        sampleNumber = length(obj.item{stream_count}.timeStamp) + buffer_counter;
                        warning('MoBILAB:corruptedFile',['File ''' drfName ''' is corrupted from sample ' num2str(sampleNumber)]);
                    end
                    for stream_count=1:N
                        try %#ok
                            % buffer_counter = length(timeStamp{stream_count});
                            timeStamp{stream_count} = timeStamp{stream_count}(1:buffer_counter-1);
                            event{stream_count}     = event{stream_count}(1:buffer_counter-1);
                            if ~isempty(samples{stream_count})
                                samples{stream_count} = samples{stream_count}(:,1:buffer_counter-1);
                                obj.item{stream_count}.addSamples(repmat(convert_to_uV{stream_count},...
                                    buffer_counter-1,1).*samples{stream_count}(:),timeStamp{stream_count},...
                                    event{stream_count});
                            else obj.item{stream_count}.addSamples([],timeStamp{stream_count}(:),event{stream_count}(:));
                            end
                        end
                    end
                end
            end
            
            % transpose the file to get the time dimension as the first
            obj.disconnect;
            for stream_count=1:N
                binFile = [obj.mobiDataDirectory filesep name{stream_count} '_' obj.item{stream_count}.uuid '_' obj.sessionUUID '.bin'];
                fid2 = fopen(binFile,'w');
                cleanupHandle2 = onCleanup(@()fclose(fid2));
                
                if obj.item{stream_count}.numberOfChannels > 0
                    tmp_mmfObj = memmapfile(obj.item{stream_count}.binFile,'Format',...
                        {obj.item{stream_count}.precision [obj.item{stream_count}.numberOfChannels length(obj.item{stream_count}.timeStamp)] 'x'},...
                        'Writable',obj.item{stream_count}.writable);
                    try data = tmp_mmfObj.Data.x; %#ok
                    catch ME
                        msg = sprintf('A subscripting operation on the Data field attempted to create a\ncomma-separated list. The memmapfile class does not support the use of\ncomma-separated lists when subscripting.');
                        if strcmp(ME.message,msg) && obj.item{stream_count}.numberOfChannels == 1
                            data = zeros(1,length(obj.item{stream_count}.timeStamp));
                            data(obj.item{stream_count}.event.latencyInFrame) = str2double(obj.item{stream_count}.event.label); %#ok
                        else
                            ME.rethrow;
                        end
                    end
                    for jt=1:obj.item{stream_count}.numberOfChannels
                        sample = eval([obj.item{stream_count}.precision '(data(jt,:));']);
                        fwrite(fid2,sample,obj.item{stream_count}.precision);
                    end
                elseif obj.item{stream_count}.numberOfChannels == 0 && ~isempty(obj.item{stream_count}.event.latencyInFrame) && ~obj.item{end}.isMemoryMappingActive
                    data =  zeros(1,length(obj.item{stream_count}.timeStamp));%#ok
                    sample = eval([obj.item{stream_count}.precision '(data(:));']);
                    fwrite(fid2,sample,obj.item{stream_count}.precision);
                    
                    clear event
                    event.hedTag = obj.item{stream_count}.event.hedTag;
                    event.label = obj.item{stream_count}.event.label;
                    event.latencyInFrame = obj.item{stream_count}.event.latencyInFrame;
                    timeStamp = obj.item{stream_count}.timeStamp; %#ok
                    numberOfChannels = 1; %#ok
                    save(obj.item{stream_count}.header,'-mat','-append','event','timeStamp','numberOfChannels');
                    
                    tmp_header = obj.item{stream_count}.header;
                    constructorHandle = eval(['@' class(obj.item{stream_count})]);
                    delete(obj.item{stream_count});
                    obj.item{stream_count} = constructorHandle(tmp_header);
                else
                    obj.item{stream_count}.mmfObj = [];
                end
                java.io.File(obj.item{stream_count}.binFile).delete();
                obj.item{stream_count}.binFile = binFile;
                [path,filename] = fileparts(obj.item{stream_count}.binFile);
                copyfile(obj.item{stream_count}.header,fullfile(path,[filename '.hdr']));
                java.io.File(obj.item{stream_count}.header).delete();
                obj.item{stream_count}.header = fullfile(path,[filename '.hdr']);
                header = obj.item{stream_count}.header; %#ok
                binFile = obj.item{stream_count}.binFile;%#ok
                clear event
                event.hedTag = obj.item{stream_count}.event.hedTag;
                event.label = obj.item{stream_count}.event.label;
                event.latencyInFrame = obj.item{stream_count}.event.latencyInFrame;
                timeStamp = obj.item{stream_count}.timeStamp; %#ok
                save(obj.item{stream_count}.header,'-mat','-append','event','timeStamp','header','binFile');
                obj.item{stream_count}.container = obj;
            end
            
            obj.connect;
            obj.checkTimeStamps;
            obj.findSpaceBoundary;
            obj.updateLogicalStructure;
            obj.save(obj.mobiDataDirectory);
            obj.container.lockGui;
        end
        %%
        function checkTimeStamps(obj)
            if obj.item{1}.timeStamp == 0, return;end
            N = length(obj.item);
            if N < 1, return;end
            disp('Correcting time stamps if needed...')
            t0 = double(obj.item{1}.timeStamp);
            ind = find(abs(diff(t0)) > 2); % find discontinuities in time longer than 2 sec
            t0 = (t0-t0(1))*1e-3;          % from mili-sec to sec 
            sf = 1./(diff(t0)+eps);            
            tnew = (0:length(t0)-1)/obj.item{1}.samplingRate;
            
            t = ttest(zscore(sf-obj.item{1}.samplingRate));
            t(isnan(t)) = 0;
            if t, warning(['There is an inconsistency between the sampling rate in the .xml file (' num2str(obj.item{1}.samplingRate) ' Hz), and the actual sampling rate (' num2str(mean(sf)) ' Hz).']);end
            for it=1:N, obj.item{it}.correctTimeStampDefects(tnew,obj.item{1}.samplingRate);end
            if ~isempty(ind), for it=1:N, obj.item{it}.event.addEvent(ind,'boundary');end;end
        end
    end
    methods(Static)
        function obj = merge(obj,dsourceObjs)
           parentCommand.commandName = 'dataSourceFromFolder';
            parentCommand.varargin = {fileparts(dsourceObjs{1}.item{1}.parentCommand.varargin{1}),obj.mobiDataDirectory};
            Nf = length(dsourceObjs);
            Nstreams = length(dsourceObjs{1}.item);
            
            offset = zeros(Nstreams,Nf);
            for it=1:Nstreams
                for folder=1:Nf
                    offset(it,folder) = dsourceObjs{folder}.item{it}.timeStamp(end);
                    if dsourceObjs{1}.item{it}.numberOfChannels ~= dsourceObjs{folder}.item{it}.numberOfChannels
                        error('MoBILAB:merging','Cannot merge streams with different number of channels!!!');
                    end
                end
            end
            offset = cumsum(max(offset));
            
            for it=1:Nstreams
                uuid = generateUUID;
                binFile = fullfile(obj.mobiDataDirectory, [dsourceObjs{1}.item{it}.name '_' uuid '_' obj.sessionUUID '.bin']);
                fid = fopen(binFile,'w');
                for ch=1:dsourceObjs{1}.item{it}.numberOfChannels
                    for folder=1:Nf, fwrite(fid,dsourceObjs{folder}.item{it}.data(:,ch),dsourceObjs{1}.item{it}.precision);end
                end
                fclose(fid);
                cumSize = zeros(Nf,1);
                time = dsourceObjs{1}.item{it}.timeStamp;
                for folder=2:Nf, time = [time offset(folder-1)+1/dsourceObjs{folder}.item{it}.samplingRate+dsourceObjs{folder}.item{it}.timeStamp];end%#ok
                
                eventObj = event;
                offsetEvents = [0;cumsum(cumSize)];
                for folder=1:Nf
                    latency = offsetEvents(folder) + dsourceObjs{folder}.item{it}.event.latencyInFrame;
                    hedTag = dsourceObjs{folder}.item{it}.event.hedTag;
                    eventObj = eventObj.addEvent(latency,hedTag);
                    if folder+1 <= Nf, eventObj = eventObj.addEvent(offsetEvents(folder+1),'boundary');end
                end
                
                metadata = saveobj(dsourceObjs{1}.item{it});
                metadata.binFile = binFile;
                metadata.sessionUUID = obj.sessionUUID;
                metadata.parentCommand = parentCommand;
                metadata.timeStamp = time;
                metadata.event = eventObj.saveobj;
                header = metadata2headerFile(metadata);
                obj.addItem(header);
            end
        end
    end
end

%%
function [data,timestamp,event,number_of_channels,precision] = read_stream(fid)
timestamp  = fread(fid,1,'int32');
event      = fread(fid,1,'int32');
number_of_channels = fread(fid,1,'int32');
item_size  = fread(fid,1,'int32');
precision  = sprintf('int%i',8*item_size);
try
    data = fread(fid,number_of_channels,precision);
catch ME
    ME.rethrow;
end
end
