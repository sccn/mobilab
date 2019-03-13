% Definition of the class dataSourceXDF. This class imports into MoBILAB files 
% recorder by the Lab Streaming Layer (LSL) library saved in xdf or xdfz format.
% 
% See more details about LSL here: https://code.google.com/p/labstreaminglayer/
% 
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Oct-2012

classdef dataSourceXDF < dataSource
    methods
        function obj = dataSourceXDF(varargin)
            % Creates a dataSourceXDF object.
            % 
            % Input arguments:
            %       file:              xdf or xdfz file recorded by LSL
            %       mobiDataDirectory: path to the directory where the collection of 
            %                          files are or will be stored. 
            %                           
            % Output arguments:          
            %       obj:               dataSource object (handle)
            %
            % Usage:
            %        % file: xdf or xdfz file recorded by LSL
            %        % mobiDataDirectory: path to the directory where the collection of
            %        %                    files are or will be stored.
            %
            %        obj = dataSourceXDF( file,  mobiDataDirectory);
            

            if nargin==1, varargin = varargin{1};end
            if length(varargin) < 2, error('Not enough input arguments.');end
            sourceFileName = varargin{1};
            mobiDataDirectory = varargin{2};
            mismatchFlag = false;
            [~,~,ext] = fileparts(sourceFileName);
            if ~any(ismember({'.xdf','.xdfz'},ext))
                error('MoBILAB:isNotXDF',['dataSourceXDF cannot read ''' ext ''' format.']);
            end
            if ~exist('load_xdf','file')
                error('MoBILAB:xdfimportMissing','xdfimport plugin is missing.')
            end
            uuid = generateUUID;
            obj@dataSource(mobiDataDirectory,uuid);
            obj.listenerHandle.Enabled = false;
            obj.checkThisFolder(mobiDataDirectory);
            logFile = [obj.mobiDataDirectory filesep 'logfile.txt'];
            fLog = fopen(logFile,'w');
            seeLogFile = false;
            parentCommand.commandName = 'dataSourceXDF';
            parentCommand.varargin{1} = sourceFileName;
            parentCommand.varargin{2} = mobiDataDirectory;
            
            try
                obj.container.lockGui('Reading...');
                streams = load_xdf(sourceFileName,'OnChunk',@subsampleChunk);
                obj.container.lockGui;
                
                rmThis = false(length(streams),1);
                for stream_count=1:length(streams)
                    if isempty(streams{stream_count}.time_stamps)
                        msg = ['Stream ' streams{stream_count}.info.name ' has no time stamps. It cannot be imported.'];
                        warning('MoBILAB:noData',msg);
                        fprintf(fLog,'%s\n',msg);
                        rmThis(stream_count) = true;
                        seeLogFile = true;
                    end
                end
                
                for stream_count=1:length(streams)
                    try %#ok
                        disp([num2str(stream_count) '-> Stream ' streams{stream_count}.info.name ':']);
                        disp(['     uuid:       ' streams{stream_count}.info.uid]);
                        disp(['     host:       ' streams{stream_count}.info.hostname]);
                        disp(['     type:       ' streams{stream_count}.info.type]);
                        disp(['     session id: ' streams{stream_count}.info.session_id]);
                        disp(['     created at: ' streams{stream_count}.info.created_at]);
                        disp(['     samples:    ' streams{stream_count}.info.sample_count]);
                        disp(['     channels:   ' num2str(size(streams{stream_count}.time_series,1))]);
                    end
                    if isempty(streams{stream_count}.time_stamps)
                        msg = ['Stream ' streams{stream_count}.info.name ' has no time stamps. It cannot be imported.'];
                        warning('MoBILAB:noData',msg);
                        fprintf(fLog,'%s\n',msg);
                        rmThis(stream_count) = true;
                        seeLogFile = true;
                    end
                end
                
                uuid_list = cell(length(streams),1);
                tmp_names = cell(length(streams),1);
                for stream_count=1:length(streams)
                    uuid_list{stream_count} = char(streams{stream_count}.info.uid);
                    tmp_names{stream_count} = lower(streams{stream_count}.info.name);
                end
                [~,loc1,~] = intersect(uuid_list,unique(uuid_list));
                rep_streams = setdiff(1:length(streams),loc1);
                if ~isempty(rep_streams)
                    warning('MoBILAB:repeatedStream',['Streams ' num2str(rep_streams) ' are repeated, their first instance will be ignored. This probably happened because there are LSL streams of the same name running in the background.']);
                    rmThis(rep_streams) = 1;
                end
               
                if any(rmThis)
                    tmp = streams(rmThis);
                    for k=1:length(tmp), if isfield(tmp{k},'tmpfile') && exist(tmp{k}.tmpfile,'file'), java.io.File(tmp{k}.tmpfile).delete();end;end
                    streams(rmThis) = [];
                end
                
                tmp_names = cell(length(streams),1);
                for stream_count=1:length(streams), tmp_names{stream_count} = lower(streams{stream_count}.info.name);end
                [~,loc] = sort(tmp_names);
                streams = streams(loc);
                stack = dbstack;
                if ~any(~cellfun(@isempty,strfind({stack.name},'dataSourceFromFolder')))
                    t0 = inf(length(streams),1);
                    for stream_count=1:length(streams), t0(stream_count) = streams{stream_count}.time_stamps(1);end
                    t0 = min(t0);
                else
                    t0 = 0;
                end
                
                % mapping to mobilab's representation
                namePool = {};
                obj.container.initStatusbar(1,length(streams),'Loading streams...');
                for stream_count=1:length(streams)
                    try
                        name = lower(streams{stream_count}.info.name);
                        if strcmpi(name,'phasespace'), name = 'mocap';end
                        if strcmpi(name,'biosemi'), name = 'eeg';end
                        name(name == ' ') = '_';
                        name = [name '_' streams{stream_count}.info.hostname]; %#ok
                        I = strfind(namePool,name);
                        I = find(~cellfun(@isempty,I));
                        if ~isempty(I), name  = [name num2str(length(I))];end %#ok
                        namePool{end+1} = name; %#ok
                        type = lower(streams{stream_count}.info.type);
                        numberOfChannels = str2double(streams{stream_count}.info.channel_count);
                        samplingRate = str2double(streams{stream_count}.info.nominal_srate);
                        % precision = streams{stream_count}.info.channel_format;
                        precision = 'double';
                        uuid = streams{stream_count}.info.uid;
                        if ~ischar(uuid), uuid = char(uuid);end
                        uuid = lower(uuid);
                        if isempty(streams{stream_count}.time_stamps), timeStamp = 0;
                        else timeStamp = streams{stream_count}.time_stamps(:)'-t0;
                        end
                        if samplingRate == 0 && length(timeStamp)
                            samplingRate = median(1./(diff(timeStamp)));
                        end
                        labels = cell(numberOfChannels,1);
                        channelType = cell(numberOfChannels,1);
                        unit = cell(numberOfChannels,1);
                        for it=1:numberOfChannels
                            try
                                if isfield(streams{stream_count}.info.desc,'channels')
                                    if isfield(streams{stream_count}.info.desc.channels,'channel') && numberOfChannels == 1 && isfield(streams{stream_count}.info.desc.channels.channel,'label')
                                        labels{it} = streams{stream_count}.info.desc.channels.channel.label;
                                        if isfield(streams{stream_count}.info.desc.channels.channel,'unit')
                                             unit{it} = streams{stream_count}.info.desc.channels.channel.unit;
                                        else unit{it} = 'unknown';
                                        end 
                                    elseif isfield(streams{stream_count}.info.desc.channels,'channel') && isfield(streams{stream_count}.info.desc.channels.channel{it},'label')
                                        labels{it} = streams{stream_count}.info.desc.channels.channel{it}.label;
                                        if isfield(streams{stream_count}.info.desc.channels.channel{it},'unit')
                                            unit{it}   = streams{stream_count}.info.desc.channels.channel{it}.unit;
                                        else unit{it} = 'unknown';
                                        end
                                    elseif isfield(streams{stream_count}.info.desc.channels,'channel') && isfield(streams{stream_count}.info.desc.channels.channel{it},'name')
                                        labels{it} = streams{stream_count}.info.desc.channels.channel{it}.name;
                                        if isfield(streams{stream_count}.info.desc.channels.channel{it},'unit')
                                            unit{it}   = streams{stream_count}.info.desc.channels.channel{it}.unit;
                                        else unit{it} = 'unknown';
                                        end
                                    elseif numberOfChannels > 1 && isfield(streams{stream_count}.info.desc.channels{it},'name')
                                        labels{it} = streams{stream_count}.info.desc.channels{it}.name;
                                        unit{it}   = streams{stream_count}.info.desc.channels{it}.unit;
                                    elseif numberOfChannels == 1 && isfield(streams{stream_count}.info.desc.channels,'name')
                                        labels{it} = streams{stream_count}.info.desc.channels.name;
                                        unit{it}   = streams{stream_count}.info.desc.channels.unit;
                                    else
                                        labels{it} = ['Unknown' num2str(it)];
                                        unit{it} = 'unknown';
                                    end
                                    if isfield(streams{stream_count}.info.desc.channels,'channel') && numberOfChannels == 1 && isfield(streams{stream_count}.info.desc.channels.channel,'type')
                                        channelType{it} = streams{stream_count}.info.desc.channels.channel.type;
                                    elseif isfield(streams{stream_count}.info.desc.channels,'channel') && numberOfChannels == 1 && ~isfield(streams{stream_count}.info.desc.channels.channel,'type')
                                        channelType{it} = type;
                                    elseif isfield(streams{stream_count}.info.desc.channels,'type')
                                        channelType{it} = streams{stream_count}.info.desc.channels.type;
                                    elseif isfield(streams{stream_count}.info.desc.channels,'channel') && isfield(streams{stream_count}.info.desc.channels.channel{it},'type')
                                        channelType{it} = streams{stream_count}.info.desc.channels.channel{it}.type;
                                    elseif numberOfChannels > 1 && isfield(streams{stream_count}.info.desc,'channels') && isfield(streams{stream_count}.info.desc.channels{it},'type')
                                        channelType{it} = streams{stream_count}.info.desc.channels{it}.type;
                                    elseif numberOfChannels == 1 && isfield(streams{stream_count}.info.desc,'channels') && isfield(streams{stream_count}.info.desc.channels,'type')
                                        channelType{it} = streams{stream_count}.info.desc.channels{it}.type;
                                    else
                                        channelType{it} = type;
                                    end
                                else
                                    labels{it}  = ['Unknown' num2str(it)];
                                    channelType{it} = type;
                                    unit{it} = 'unknown';
                                end
                            catch ME
                                if ~mismatchFlag
                                    warning(ME.identifier,ME.message);
                                    msg = ['Wrong structure or missing fields in stream "' name '".'];
                                    warning('MoBILAB:mismatch',msg)
                                    fprintf(fLog,'%s\n',msg);
                                    disp('Doing my best to fix this...');
                                    mismatchFlag = true;
                                    seeLogFile = true;
                                end
                                labels{it}  = ['Unknown' num2str(it)];
                                channelType{it} = type;
                                unit{it} = 'unknown';
                            end
                        end
                        hardwareMetaDataObj = streams{stream_count}.info;
                        clear metadata
                        eventStruct = event;
                        metadata = struct('binFile','','header','','name',name,'timeStamp',timeStamp,'numberOfChannels',numberOfChannels,'precision',precision,...
                            'uuid',uuid,'sessionUUID',obj.sessionUUID,'writable',false,'unit',[],'hardwareMetaData',hardwareMetaDataObj,...
                            'parentCommand',parentCommand,'label',[],'event',eventStruct.saveobj,'notes','','samplingRate',samplingRate);
                        metadata.unit = unit;
                        metadata.label = labels;
                        
                        % eeg
                        if strcmpi(streams{stream_count}.info.type,'eeg')
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            channelSpace = nan(numberOfChannels,3);
                            for ch=1:numberOfChannels
                                try %#ok
                                    channelSpace(ch,:) = [str2double(streams{stream_count}.info.desc.channels.channel{ch}.location.X) ...
                                        str2double(streams{stream_count}.info.desc.channels.channel{ch}.location.Y) ...
                                        str2double(streams{stream_count}.info.desc.channels.channel{ch}.location.Z)];
                                end
                            end
                            eegChannels = ismember(lower(channelType),'eeg');
                            % mmfObj = memmapfile(streams{stream_count}.tmpfile,'Format',{streams{stream_count}.precision...
                            %     [length(eegChannels) length(streams{stream_count}.time_stamps)] 'x'},'Writable',false);
                            if any(~eegChannels)
                                auxChannel.label = labels(~eegChannels);
                                % auxChannel.data = mmfObj.Data.x(~eegChannels,:)';
                                auxChannel.data = eval([precision '( streams{stream_count}.time_series(~eegChannels,:))'])';
                                %streams{stream_count}.time_series(~eegChannels,:) = [];
                            else
                                auxChannel.label = {};
                                auxChannel.data = [];
                            end
                            labels(~eegChannels) = [];
                            channelSpace(~eegChannels,:) = [];
                            unit(~eegChannels) = [];
                            noLoc = find(isnan(channelSpace(:,1)));
                            channels2write = find(eegChannels);
                            if length(noLoc) < length(channels2write)
                                channels2write(noLoc) = [];
                                labels(noLoc) = [];
                                channelSpace(noLoc,:) = [];
                                if ~isempty(noLoc) && length(noLoc) ~= numberOfChannels
                                    msg = sprintf('In %s the following sensors have no location: %s.\n',name,num2str(noLoc(:)'));
                                    msg = sprintf('%sMoBILAB will not import them, contact the support team for more information.\n',msg);
                                    fprintf(fLog,'%s',msg);
                                    seeLogFile = true;
                                    warning(msg); %#ok
                                end
                            end
                            numberOfChannels = length(labels);
                            for ch=1:numberOfChannels, if isempty(labels{ch}), labels{ch} = ['Unknown' num2str(ch)];end;end
                            reference = [];
                            isReferenced = false;
                            try reference = find(ismember(labels,streams{stream_count}.info.desc.reference.label));end %#ok
                            try isReferenced = ~strcmpi(streams{stream_count}.info.desc.reference.subtracted,'no');end %#ok
                            try
                                Nfch = length(streams{stream_count}.info.desc.fiducials.fiducial);
                                for fch=1:Nfch
                                    if ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'fidnz')) || ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'nasion'))
                                        fiducials.nasion = [str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.X) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Y) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Z)];
                                    elseif ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'fidt9')) || ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'lpa'))
                                        fiducials.lpa = [str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.X) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Y) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Z)];
                                    elseif ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'fidt10')) || ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'rpa'))
                                        fiducials.rpa = [str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.X) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Y) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Z)];
                                    elseif ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'fidt11')) || ~isempty(strfind(streams{stream_count}.info.desc.fiducials.fiducial{fch}.label,'vertex'))
                                        fiducials.vertex = [str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.X) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Y) ...
                                            str2double(streams{stream_count}.info.desc.fiducials.fiducial{fch}.location.Z)];
                                    end
                                end
                            catch
                                fiducials = [];
                            end
                            fid = fopen(binFile,'w');
                            fwrite(fid,streams{stream_count}.time_series(channels2write,:)',precision);
                            fclose(fid);
                            metadata.numberOfChannels = numberOfChannels;
                            metadata.label = labels;
                            metadata.binFile = binFile;
                            metadata.hardwareMetaData = hardwareMetaDataObj;
                            metadata.channelSpace = channelSpace;
                            metadata.reference = reference;
                            metadata.isReferenced = isReferenced;
                            metadata.fiducials = fiducials;
                            metadata.auxChannel = auxChannel;
                            metadata.unit = unit;
                            metadata.class = 'eeg';
                            header = metadata2headerFile(metadata);
                            
                        % mocap    
                        elseif any(ismember({'mocap' 'control'},lower(streams{stream_count}.info.type))) && isempty(strfind(lower(streams{stream_count}.info.name),'wii'))
                            if strcmp(streams{stream_count}.info.name,'PhaseSpace') && ~isempty(channelType)
                                ind = ~cellfun(@isempty,strfind(channelType,'Position'));
                            elseif ~isempty(channelType)
                                ind = ~cellfun(@isempty,strfind(channelType,'Position'));
                            else ind = true(numberOfChannels,1);
                            end
                            channels2write = 1:numberOfChannels;
                            
                            unit(~ind) = [];
                            channels2write(~ind) = [];
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            auxChannel.label = metadata.label(~ind);
                            auxChannel.data = streams{stream_count}.time_series(~ind,:)';
                            fid = fopen(binFile,'w');
                            fwrite(fid,streams{stream_count}.time_series(channels2write,:)',precision);
                            fclose(fid);
                            % clear mmfObj
                            metadata.label(~ind) = [];
                            numberOfChannels = length(metadata.label);
                            metadata.numberOfChannels = numberOfChannels;
                            metadata.label = labels(channels2write);
                            metadata.binFile = binFile;
                            metadata.animationParameters = struct('limits',[],'conn',[],'bodymodel',[]);
                            metadata.unit = unit;
                            metadata.class = 'mocap';
                            metadata.auxChannel = auxChannel;
                            header = metadata2headerFile(metadata);
                        
                        % audiocontrol    
                        elseif ~isempty(strfind(lower(streams{stream_count}.info.type),'audiocontrol'))
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            metadata.writable = true;
                            metadata.numberOfChannels = 1;
                            metadata.label = {'audiocontrol'};
                            fid2 =  fopen(binFile,'w');
                            fwrite(fid2,zeros(length(streams{stream_count}.time_stamps),1),'int16');
                            fclose(fid2);
                            
                            Ntags = length(streams{stream_count}.time_series);
                            hedTag = cell(Ntags,1);
                            enable = {'onset','offset'};
                            for tag=1:Ntags
                                if str2double(streams{stream_count}.time_series{8,tag})
                                    hedTag{tag} = ['Stimulus/Auditory/File/' streams{stream_count}.time_series{1,tag} ',Stimulus/Auditory/' enable{str2double(streams{stream_count}.time_series{2,tag})} ',Stimulus/Auditory/Volume/'...
                                        streams{stream_count}.time_series{3,tag} ',Stimulus/Auditory/Azimuth/' streams{stream_count}.time_series{4,tag} '-degrees,Stimulus/Auditory/Elevation/' streams{stream_count}.time_series{5,tag}...
                                        '-degrees,Stimulus/Auditory/Spread/' streams{stream_count}.time_series{6,tag} '-degrees,Stimulus/Auditory/Speed/' streams{stream_count}.time_series{7,tag} ',Stimulus/Auditory/Looping'];
                                else
                                    hedTag{tag} = ['Stimulus/Auditory/File/' streams{stream_count}.time_series{1,tag} ',Stimulus/Auditory/' enable{str2double(streams{stream_count}.time_series{2,tag})} ',Stimulus/Auditory/Volume/'...
                                        streams{stream_count}.time_series{3,tag} ',Stimulus/Auditory/Azimuth/' streams{stream_count}.time_series{4,tag} '-degrees,Stimulus/Auditory/Elevation/' streams{stream_count}.time_series{5,tag}...
                                        '-degrees,Stimulus/Auditory/Spread/' streams{stream_count}.time_series{6,tag} '-degrees,Stimulus/Auditory/Speed/' streams{stream_count}.time_series{7,tag}];
                                end
                            end
                            eventObj = event;
                            eventObj = eventObj.addEvent(1:length(streams{stream_count}.time_stamps),hedTag);
                            metadata.binFile = binFile;
                            metadata.precision = 'int16';
                            metadata.event = eventObj.saveobj;
                            metadata.class = 'markerStream';
                            header = metadata2headerFile(metadata);
                        
                        % markers    
                        elseif ~isempty(strfind(lower(streams{stream_count}.info.type),'marker')) || ~isempty(strfind(lower(streams{stream_count}.info.type),'event'))
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            metadata.writable = true;
                            fid2 =  fopen(binFile,'w');
                            fwrite(fid2,zeros(length(streams{stream_count}.time_stamps),1),'int16');
                            fclose(fid2);
                            
                            eventObj = event;
                            if isnumeric(streams{stream_count}.time_series)
                                Nev = length(streams{stream_count}.time_series);
                                eventLabel = cell(Nev,1);
                                for ev_k=1:Nev
                                    eventLabel{ev_k} = num2str(streams{stream_count}.time_series(ev_k));
                                end
                            else
                                eventLabel = streams{stream_count}.time_series;
                            end
                            eventObj = eventObj.addEvent(1:length(streams{stream_count}.time_stamps),eventLabel);
                            metadata.binFile = binFile;
                            metadata.precision = 'int16';
                            metadata.event = eventObj.saveobj;
                            metadata.class = 'markerStream';
                            header = metadata2headerFile(metadata);
                            
                        % video
                        elseif ~isempty(strfind(lower(streams{stream_count}.info.type),'videostream')) ||...
                                ~isempty(strfind(lower(streams{stream_count}.info.type),'video'))
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            fid = fopen(binFile,'w');
                            fwrite(fid,streams{stream_count}.time_series',precision);
                            fclose(fid);
                            metadata.binFile = binFile;
                            try
                                exist(streams{stream_count}.info.desc.videoFile,'file')
                                copyfile(streams{stream_count}.info.desc.videoFile, mobiDataDirectory)
                                [~,filename,ext] = fileparts(streams{stream_count}.info.desc.videoFile);
                                metadata.videoFile = fullfile(mobiDataDirectory, [filename,ext]);
                            catch
                                metadata.videoFile = '';
                            end
                            
                            metadata.unit = 'none';
                            metadata.class = 'videoStream';
                            header = metadata2headerFile(metadata);
                            
                        % scene    
                        elseif strfind(lower(streams{stream_count}.info.type),'scenestream')
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            % mmfObj = memmapfile(streams{stream_count}.tmpfile,'Format',{streams{stream_count}.precision...
                            %     [numberOfChannels length(streams{stream_count}.time_stamps)] 'x'},'Writable',false);
                            fid = fopen(binFile,'w');
                            % for ch=1:numberOfChannels, fwrite(fid,mmfObj.Data.x(ch,:)',precision);end
                            fwrite(fid,streams{stream_count}.time_series',precision);
                            fclose(fid);
                            % clear mmfObj
                            metadata.binFile = binFile;
                            metadata.class = 'sceneStream';
                            header = metadata2headerFile(metadata);
                        
                        % audio
                        elseif strfind(lower(streams{stream_count}.info.type),'audio')
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            fid = fopen(binFile,'w');
                            fwrite(fid,streams{stream_count}.time_series',precision);
                            fclose(fid);
                            metadata.binFile = binFile;
                            metadata.class = 'audioStream';
                            header = metadata2headerFile(metadata);
                            
                        % hotspot
                        elseif strfind(lower(streams{stream_count}.info.type),'hotspotdata')
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            % mmfObj = memmapfile(streams{stream_count}.tmpfile,'Format',{streams{stream_count}.precision...
                            %     [numberOfChannels length(streams{stream_count}.time_stamps)] 'x'},'Writable',false);
                            fid = fopen(binFile,'w');
                            % for ch=1:numberOfChannels, fwrite(fid,mmfObj.Data.x(ch,:)',precision);end
                            fwrite(fid,streams{stream_count}.time_series',precision);
                            fclose(fid);
                            % clear mmfObj
                            metadata.binFile = binFile;
                            metadata.class = 'gazeStream';
                            header = metadata2headerFile(metadata);
                            
                        else
                            binFile = [obj.mobiDataDirectory filesep name '_' uuid '_' obj.sessionUUID '.bin'];
                            % mmfObj = memmapfile(streams{stream_count}.tmpfile,'Format',{streams{stream_count}.precision...
                            %     [numberOfChannels length(streams{stream_count}.time_stamps)] 'x'},'Writable',false);
                            fid = fopen(binFile,'w');
                            % for ch=1:numberOfChannels, fwrite(fid,mmfObj.Data.x(ch,:)',precision);end
                            fwrite(fid,streams{stream_count}.time_series',precision);
                            fclose(fid);
                            % clear mmfObj
                            metadata.binFile = binFile;
                            metadata.class = 'dataStream';
                            header = metadata2headerFile(metadata);
                        end
                        obj.addItem(header);
                        if isfield(streams{stream_count},'tmpfile'), java.io.File(streams{stream_count}.tmpfile).delete();end
                        streams{stream_count}.time_series = [];
                        streams{stream_count}.time_stamps = [];
                        mismatchFlag = false;
                    catch ME
                        fprintf(fLog,'%s. \n',ME.message);
                        seeLogFile = true;
                        warning(ME.message);
                    end
                    obj.container.statusbar(stream_count);
                end
            catch ME
                fprintf(fLog,'%s\n.',ME.message);
                seeLogFile = true;
                obj.container.lockGui;
                obj.disconnect;
                for k=1:length(streams), if isfield(streams{k},'tmpfile') && exist(streams{k}.tmpfile,'file'), java.io.File(streams{k}.tmpfile).delete();end;end
                if seeLogFile, disp(['Logs were saved in: ' logFile]);end
                ME.rethrow;
            end
            for k=1:length(streams), if isfield(streams{k},'tmpfile') && exist(streams{k}.tmpfile,'file'), java.io.File(streams{k}.tmpfile).delete();end;end
            obj.listenerHandle.Enabled = true;
            obj.connect;
            obj.updateLogicalStructure;
%             stack = dbstack;
%             if ~any(~cellfun(@isempty,strfind({stack.name},'dataSourceFromFolder')))
%                 markerIndices = obj.getItemIndexFromItemClass('markerStream');
%                 if length(markerIndices) > 1, obj.makeMultiMarkerStreamObject;end
%             end
            if seeLogFile, disp(['Logs were saved in: ' logFile]);end
        end
    end
    methods(Static)
        function obj = merge(obj,dsourceObjs)
            parentCommand.commandName = 'dataSourceFromFolder';
            parentCommand.varargin = {fileparts(dsourceObjs{1}.item{1}.parentCommand.varargin{1}),obj.mobiDataDirectory};
            Nf = length(dsourceObjs);
            
            for folder=1:Nf
                for it=1:length(dsourceObjs{folder}.item)
                    try
                        sourceId = dsourceObjs{folder}.item{it}.hardwareMetaData.source_id;
                        if ~ischar(sourceId), sourceId = '';end
                        hostname = dsourceObjs{folder}.item{it}.hardwareMetaData.hostname;
                        if ~ischar(hostname), hostname = '';end
                        index    = obj.findItem(dsourceObjs{folder}.item{it}.uuid);
                        ind      = intersect(obj.getItemIndexFromSourceId(sourceId), obj.getItemIndexFromHostname(hostname));
                        if isempty(ind), ind = 0;end
                        
                        if index
                            fid = fopen(obj.item{index}.binFile,'a');
                            fwrite(fid,dsourceObjs{folder}.item{it}.data(:),obj.item{index}.precision);
                            fclose(fid);
                            
                            if ~isempty(dsourceObjs{folder}.item{it}.event.latencyInFrame)
                                latency = dsourceObjs{folder}.item{it}.event.latencyInFrame;
                                latency = length(obj.item{index}.timeStamp) + latency;
                                hedTag  = dsourceObjs{folder}.item{it}.event.hedTag;
                                obj.item{index}.event = obj.item{index}.event.addEvent(latency,hedTag);
                            end
                            obj.item{index}.event = obj.item{index}.event.addEvent(length(obj.item{index}.timeStamp),'boundary');
                            
                            saveProperty(obj.item{index},'timeStamp',[obj.item{index}.timeStamp dsourceObjs{folder}.item{it}.timeStamp]);
                            header = obj.item{index}.header;
                            constructorHandle = eval(['@' class(obj.item{index})]);
                            delete(obj.item{index});
                            obj.item{index} = constructorHandle(header);
                            obj.item{index}.container = obj;
                            
                        elseif ind
                            fid = fopen(obj.item{ind}.binFile,'a');
                            fwrite(fid,dsourceObjs{folder}.item{it}.data(:),obj.item{ind}.precision);
                            fclose(fid);
                            
                            if ~isempty(dsourceObjs{folder}.item{it}.event.latencyInFrame)
                                latency = dsourceObjs{folder}.item{it}.event.latencyInFrame;
                                latency = length(obj.item{ind}.timeStamp) + latency;
                                hedTag  = dsourceObjs{folder}.item{it}.event.hedTag;
                                obj.item{ind}.event = obj.item{ind}.event.addEvent(latency,hedTag);
                            end
                            obj.item{ind}.event = obj.item{ind}.event.addEvent(length(obj.item{ind}.timeStamp),'boundary');
                            
                            saveProperty(obj.item{ind},'timeStamp',[obj.item{ind}.timeStamp dsourceObjs{folder}.item{it}.timeStamp]);
                            header = obj.item{ind}.header;
                            constructorHandle = eval(['@' class(obj.item{ind})]);
                            delete(obj.item{ind});
                            obj.item{ind} = constructorHandle(header);
                            obj.item{ind}.container = obj;
                            
                        else
                            metadata = saveobj(dsourceObjs{folder}.item{it});
                            binFile = fullfile(obj.mobiDataDirectory, [metadata.name '_' metadata.uuid '_' obj.sessionUUID '.bin']);
                            copyfile(dsourceObjs{folder}.item{it}.binFile,binFile);
                            metadata.binFile = binFile;
                            metadata.sessionUUID = obj.sessionUUID;
                            metadata.parentCommand = parentCommand;
                            header = metadata2headerFile(metadata);
                            obj.addItem(header);
                            
                        end
                    catch ME
                        binFile = metadata.binFile;
                        header  = [binFile(1:end-3) 'hdr'];
                        if xor(exist(binFile,'file'),exist(binFile,'file'))
                            try delete(binFile);end %#ok
                            try delete(header); end %#ok
                        end
                        warning(ME.message)
                    end
                end
            end
            N = length(obj.item);
            t0 = inf(N,1);
            for it=1:N, t0(it) = obj.item{it}.timeStamp(1);end
            t0 = min(t0);
            for it=1:N, obj.item{it}.correctTimeStampDefects(obj.item{it}.timeStamp - t0);end
        end
    end
end
%%
function [values,timestamps,stream] = subsampleChunk(values,timestamps,stream,~)
try %#ok
    if str2double(stream.info.nominal_srate) > 2e4
        values = values(:,1:2:end);
        timestamps = timestamps(1:2:end);
    end
end
end