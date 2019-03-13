% Definition of the class eeg. This class defines analysis methods
% exclusively for EEG data.
%
% For more details visit: https://code.google.com/p/mobilab/ 
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Apr-2011

classdef eeg < dataStream
    properties
        isReferenced    % Boolean that reflects whether an EEG data set has been re-referenced of not.
        reference       % Cell array with the labels of the channels used to compute the reference.
        channelSpace
        fiducials
    end
    
    methods
        function obj = eeg(header)
            % Creates an eeg object.
            % 
            % Input arguments:
            %       header: header file (string)
            %       
            % Output arguments:
            %       obj:    eeg object (handle)
            % 
            % Usage:
            %       obj = eeg(header);

            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        %%
        function channelSpace = get.channelSpace(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.channelSpace')), label = obj.channelSpace;return;end
            if isempty(obj.channelSpace), obj.channelSpace = retrieveProperty(obj,'channelSpace');end
            channelSpace = obj.channelSpace;
        end
        function set.channelSpace(obj,channelSpace)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.get.channelSpace'))
                obj.channelSpace = channelSpace;
                return;
            end
            obj.channelSpace = channelSpace;
            saveProperty(obj,'channelSpace',channelSpace);
        end
        %%
        function isReferenced = get.isReferenced(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.set.isReferenced'))
                isReferenced = obj.isReferenced;
                return;
            end
            if isempty(obj.isReferenced)
                try obj.isReferenced = retrieveProperty(obj,'isReferenced');
                catch
                    isReferenced = false;
                    save(obj.header,'-mat','-append','isReferenced')
                    obj.isReferenced = isReferenced;
                end
            end
            isReferenced = obj.isReferenced;
        end
        function set.isReferenced(obj,isReferenced)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.get.isReferenced'))
                obj.isReferenced = isReferenced;
                return;
            end
            obj.isReferenced = isReferenced;
            saveProperty(obj,'isReferenced',isReferenced)
        end
        %%
        function reference = get.reference(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.set.reference'))
                reference = obj.reference;
                return;
            end
            if isempty(obj.reference), obj.reference = retrieveProperty(obj,'reference');end
            reference = obj.reference;
        end
        function set.reference(obj,reference)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.get.reference'))
                obj.reference = reference;
                return;
            end
            obj.reference = reference;
            saveProperty(obj,'reference',reference);
        end
        %%
        function jsonObj = serialize(obj)
            metadata = saveobj(obj);
            metadata.size = size(obj);
            metadata.event = obj.event.uniqueLabel;
            metadata.writable = double(metadata.writable);
            metadata.history = obj.history;
            metadata.sessionUUID = char(obj.sessionUUID);
            metadata.uuid = char(obj.uuid);
            if ~isempty(metadata.channelSpace),  metadata.hasChannelSpace = 'yes'; else metadata.hasChannelSpace = 'no';end
            metadata = rmfield(metadata,{'parentCommand' 'timeStamp' 'hardwareMetaData' 'channelSpace' 'fiducials'});
            jsonObj = savejson('',metadata,'ForceRootName', false);
        end
        %%
        function readMontage(obj,file)
            % Reads a file containing the sensor positions and its labels. 
            % If recorder, it also extracts fiducial landmarks. If the number
            % of channels in the file are less than the number of channels in
            % the object, a new object is created with the common set. If more
            % than one file is supplied the program creates separate objects 
            % containing the different set of channels selected by each file.
            % 
            % Input argument: 
            %       file: filename. The following formats are supported: 
            %             BESA or EGI3D cartesian .sfp, Polhemus .elp, 
            %             Matlab .xyz or .sph, EEGLAB .loc, and Neuroscan
            %             .asc or .dat.

            if nargin < 2
                [filename, pathname] = uigetfile2({'*.sfp','BESA or EGI 3-D cartesian files (*.sfp)';'*.elp','Polhemus native files (*.elp)';...
                    '*.xyz','Matlab xyz files (*.xyz)';'*.loc','EEGLAB polar files (*.loc)';'*.sph','Matlab spherical files (*.sph)';...
                    '*.asc','Neuroscan polar files (*.asc)';'*.dat','Neuroscan 3-D files (*.dat)';'*.*','All files (*.*)'},...
                    'Select a file containing sensor positions','MultiSelect', 'on');
                if isnumeric(filename) || isnumeric(pathname), return;end
            elseif ~ischar(file)
                [filename, pathname] = uigetfile2({'*.sfp','BESA or EGI 3-D cartesian files (*.sfp)';'*.elp','Polhemus native files (*.elp)';...
                    '*.xyz','Matlab xyz files (*.xyz)';'*.loc','EEGLAB polar files (*.loc)';'*.sph','Matlab spherical files (*.sph)';...
                    '*.asc','Neuroscan polar files (*.asc)';'*.dat','Neuroscan 3-D files (*.dat)';'*.*','All files (*.*)'},...
                    'Select a file containing sensor positions','MultiSelect', 'on');
                if isnumeric(filename) || isnumeric(pathname), return;end
            else
                [pathname,filename,ext] = fileparts(file);
                filename = [filename,ext];
            end
            
            if iscellstr(filename)
                N = length(filename);
                newChannelSpace = cell(N,1);
                newLabels = cell(N,1);
                newEEGchannel = cell(N,1);
                fiducials = cell(N,1);
                
                tmpLabel = [];
                indEEGchannels = false(obj.numberOfChannels,1);
                for it=1:obj.numberOfChannels
                    ind = obj.label{it} == '_';
                    if any(ind)
                        ind = find(ind);
                        tmpLabel{end+1} = obj.label{it}(ind(end)+1:end); %#ok
                        indEEGchannels(it) = true;
                    end
                end
                indEEGchannels = find(indEEGchannels);
                if isempty(indEEGchannels)
                    indEEGchannels = 1:obj.numberOfChannels;
                    tmpLabel = obj.label;
                end
                for it=1:N
                    file = fullfile(pathname,filename{it});
                    [eloc, labels] = readlocs( file);
                    eloc = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
                    Nl = length(labels);
                    elecIndices = false(Nl,1);
                    channelIndices = zeros(Nl,1);
                    for jt=1:Nl
                        if ~isempty(strfind(labels{jt},'fidnz')) || ~isempty(strfind(labels{jt},'nasion')) || ~isempty(strfind(labels{jt},'Nz')) 
                            fiducials{it}.nasion = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt9')) || ~isempty(strfind(labels{jt},'lpa'))
                            fiducials{it}.lpa = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt10')) || ~isempty(strfind(labels{jt},'rpa'))
                            fiducials{it}.rpa = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt10')) || ~isempty(strfind(labels{jt},'vertex'))
                            fiducials{it}.vertex = eloc(jt,:);    
                        else
                            I = find(strcmpi(tmpLabel,labels{jt}));
                            if ~isempty(I)
                                elecIndices(jt) = true;
                                channelIndices(jt) = I(1);
                            end
                        end
                    end
                    channelIndices(channelIndices==0) = [];
                    
                    newLabels{it} = labels(elecIndices);
                    newChannelSpace{it} = eloc(logical(elecIndices),:);
                    newEEGchannel{it} = indEEGchannels(channelIndices);
                    tmpLabel(channelIndices) = []; %#ok
                    indEEGchannels(channelIndices) = [];
                end
                % these two loops can be fused, however I've split them for debugging
                for it=1:N
                    cobj = divideStreamObject(obj,newEEGchannel{it},newLabels{it},[obj.name '_' num2str(it)]);
                    cobj.channelSpace = newChannelSpace{it};
                    saveProperty(cobj,'channelSpace',cobj.channelSpace);
                    try
                        cobj.fiducials = fiducials{it};
                        saveProperty(cobj,'fiducials',cobj.fiducials);
                    catch ME
                        warning(ME.message);
                        warning('MoBILAB:noFiducials','Fiducial landmarks are missing.');
                    end
                end
            else
                file = fullfile(pathname,filename);
                [eloc, labels, theta, radius, indices] = readlocs( file); %#ok
                eloc = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
                rmThis = [];
                I = strcmpi(labels,'fidnz') | strcmpi(labels,'nasion') | strcmpi(labels,'Nz'); 
                if any(I), fiducials.nasion = eloc(I,:);rmThis(1) = find(I);end
                
                I = strcmpi(labels,'fidt9') | strcmpi(labels,'lpa');     
                if any(I), fiducials.lpa = eloc(I,:);rmThis(2) = find(I);end
                
                I = strcmpi(labels,'fidt10') | strcmpi(labels,'rpa');    
                if any(I), fiducials.rpa = eloc(I,:);rmThis(3) = find(I);end
                
                I = strcmpi(labels,'vertex');
                if any(I), fiducials.vertex = eloc(I,:);rmThis(4) = find(I);end
                
                % fixing sccn old protocol bug
                for it=1:length(labels), if ~isempty(strfind(labels{it},'EXG')), labels{it}(strfind(labels{it},'EXG'):3) = 'EXT';end;end
                
                labels(rmThis) = [];
                eloc(rmThis,:) = [];
                
                if obj.numberOfChannels > length(labels) 
                    ind = obj.container.getItemIndexFromItemNameSimilarTo(obj.name);
                    tmpObj = divideStreamObject(obj,1:length(labels) ,labels,[obj.name '_' num2str(length(ind))]);
                else
                    tmpObj = obj;
                end
                tmpObj.label = labels;
                tmpObj.channelSpace = eloc;
                saveProperty(tmpObj,'channelSpace',tmpObj.channelSpace);
                if exist('fiducials','var')
                    tmpObj.fiducials = fiducials;
                    saveProperty(tmpObj,'fiducials',tmpObj.fiducials);
                else
                    warning('MoBILAB:noFiducials','Fiducial landmarks are missing.');
                end
            end
            disp('Done!!!')
        end
        %%
        function cobj = filter(obj,varargin)
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            cobj = filter@dataStream(obj,varargin);
            if isempty(cobj), return;end
            cobj.mmfObj.Writable = true;
            obj.initStatusbar(0,1,'Centering channels...');
            cobj.mmfObj.Data.x = bsxfun(@minus,cobj.mmfObj.Data.x,mean(cobj.mmfObj.Data.x));
            obj.statusbar(1);
            cobj.mmfObj.Writable = false;
        end
        %%
        function cobj = reReference(obj,channels2BeReferenced,channels2BeAveraged)
            % Re-reference the data.
            % 
            % Input arguments:
            %       channels2BeReferenced: cell array with the label of the
            %                              channels to be referenced
            %       channels2BeAveraged:   cell array with the label of the
            %                              channels to be averaged
            %          
            % Output arguments:
            %       cobj: handle to the new object
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       channels2BeReferenced = eegObj.label;
            %       channels2BeAveraged   = eegObj.label;
            %       eegObjRef = eegObj.reReference(channels2BeReferenced,channels2BeAveraged);
            
            if nargin < 2, error('Not enough input arguments.');end
            dispCommand = false;
            if isnumeric(channels2BeReferenced) && channels2BeReferenced==-1
                cobj = [];
                channels2BeReferenced = obj.label;
                channels2BeAveraged = obj.label;
                dispCommand = true;
            end
            if ~iscellstr(channels2BeReferenced), error('First argument must be a cell with the labels for the channels to be referenced.');end
            if ~iscell(channels2BeAveraged), error('Second argument must be a cell with the labels for the channels to be averaged (The ones to be taken as a reference).');end
            
            try
                ind1 = ismember(obj.label,channels2BeReferenced);
                ind2 = ismember(obj.label,channels2BeAveraged);
                
                if sum(ind1)*sum(ind2)==0, error('Some of the labels you''ve entered don''t match the labels in the object.');end
                
                ind1 = sort(find(ind1));
                ind2 = sort(find(ind2));
                
                commandHistory.commandName = 'reReference';
                commandHistory.uuid = obj.uuid;
                commandHistory.varargin{1} = obj.label(ind1);
                commandHistory.varargin{2} = obj.label(ind2);
                
                cobj = obj.copyobj(commandHistory);
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                data = obj.mmfObj.Data.x;
                ref = mean(data(:,ind2),2);
                obj.initStatusbar(1,cobj.numberOfChannels,'Re-referencing...');
                for it=1:cobj.numberOfChannels
                    cobj.mmfObj.Data.x(:,it) = data(:,ind1(it))-ref;
                    obj.statusbar(it);
                end
                cobj.isReferenced = true;
                cobj.reference = channels2BeAveraged;
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        
        %%
        function EEG = EEGstructure(obj,ismmf, passData)
            % Returns the equivalent EEG (EEGLAB) structure.
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       EEG    = eegObj.EEGstructure;
            
            if nargin < 2, ismmf = false;end
            if nargin < 3, passData = true;end
            s = dbstack;
            
            isCalledFromGui = any(~cellfun(@isempty,strfind({s.name},'myDispatch')));
            if isCalledFromGui
                disp('Running:');
                disp(['  EEG = mobilab.allStreams.item{ ' num2str(obj.container.findItem(obj.uuid)) ' }.EEGstructure;']);
            end
            
            EEG = eeg_emptyset;
            EEG.setname = obj.name;
            EEG.times = obj.timeStamp*1e3;     % from seconds to mili-seconds
            EEG.nbchan = obj.numberOfChannels;
            EEG.pnts = length(obj.timeStamp);
            EEG.trials = 1;
            EEG.srate = obj.samplingRate;
            EEG.xmin = obj.timeStamp(1);
            EEG.xmax = obj.timeStamp(end);
            
            [path,name,ext] = fileparts(obj.binFile); %#ok
            if passData
                if ismmf
                    EEG.data = [path filesep name '.fdt'];
                    EEG.filepath = path;
                    EEG.filename = name;
                    copyMMF(EEG,obj);
                else EEG.data = single(obj.mmfObj.Data.x)';
                end
            end
            
            EEG.chaninfo.nosedir = '+X';
            if isfield(obj.hardwareMetaData,'desc'), EEG.etc.desc = obj.hardwareMetaData.desc;end
                                               
            chanlocs = repmat(struct('labels',[],'type',[],'X',[],'Y',[],'Z',[],'radius',[],'theta',[]),EEG.nbchan,1);
            labels = obj.label;
            if ~isempty(obj.channelSpace)
                xyz = obj.channelSpace;
                xyz = bsxfun(@minus,xyz,mean(xyz));
                xyz = bsxfun(@rdivide,xyz,max(xyz))/2;
                for it=1:obj.numberOfChannels
                    chanlocs(it).labels = labels{it};
                    try chanlocs(it).type = obj.hardwareMetaData.desc.channels.channel{it}.type;
                    catch chanlocs(it).type = 'EEG';
                    end
                    chanlocs(it).X = xyz(it,1);
                    chanlocs(it).Y = -xyz(it,2);
                    chanlocs(it).Z = xyz(it,3);
                    [chanlocs(it).theta,chanlocs(it).radius] = cart2pol(chanlocs(it).X, chanlocs(it).Y, chanlocs(it).Z);
                    chanlocs(it).theta = chanlocs(it).theta*180/pi;
                end
                EEG.chanlocs = chanlocs;
            end
            EEG.etc.mobi.sessionUUID = obj.sessionUUID;
            
            try 
                ALLEEG = evalin('base','ALLEEG');
            catch
                ALLEEG = [];
            end
                        
            if isempty(obj.event.label)
                if ismmf, pop_saveset( EEG, [name '.set'],path);end
                if isCalledFromGui
                    [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);
                    assignin('base','ALLEEG',ALLEEG);
                    assignin('base','CURRENTSET',CURRENTSET);
                    assignin('base','EEG',EEG);
                    try
                        evalin('base','eeglab redraw');
                    catch ME
                        disp(ME.message)
                    end
                end
                return;
            end
            
            type = obj.event.label;
            latency = obj.event.latencyInFrame;
            loc = ismember(type,'boundary');
            if sum(loc)
                loc = find(loc);
                [~,loc2] = unique(latency(loc));
                [~,rmloc] = setdiff(loc,loc(loc2));
                latency(loc(rmloc)) = [];
                type(loc(rmloc)) = [];
            end
            
            if ~isempty(latency) && ~isempty(type)
                [latency,loc] = sort(latency,'ascend');
                type = type(loc); 
                EEG.event = repmat(struct('type','','latency',0,'duration',0,'urevent',1),1,length(latency));
                for it=1:length(latency)
                    EEG.event(it).type = type{it};
                    EEG.event(it).latency = latency(it);
                    EEG.event(it).urevent = it;
                    if strcmp(type,'boundary'), EEG.event(it).duration = NaN;end
                end
                EEG.urevent = EEG.event;
            end
            if ismmf && passData, pop_saveset( EEG, [name '.set'],path);end
            if ~isCalledFromGui
                [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);
                assignin('base','ALLEEG',ALLEEG);
                assignin('base','CURRENTSET',CURRENTSET);
                assignin('base','EEG',EEG);
                evalin('base','eeglab redraw');
            end
        end
        %%
        function epochObj = epoching(obj,eventLabelOrLatency, timeLimits, channels, condition,subjectID,preStimulusLatency)
            % Creates epoch objects. Epoch objects don't result in a new addition to the tree.
            % Even when the epoch object manages its data through a memory mapped file they are
            % regarded as temporal variables, once the object is destroyed the associated binary
            % file is deleted.
            %
            % Input arguments:
            %       eventLabelOrLatency: if is a string or a cell array of strings it is used as
            %                            the event around to make the trials; if is a vector is
            %                            interpreted as the sample number of the events  around
            %                            to make the trials.
            %       timeLimits:          two elements vector specifying the size of the trial 
            %                            taking as reference the target latency. Ex: [-1 2]
            %                            correspond to one second before and two seconds after
            %                            the target event/latency.
            %       channels:            indices of the channels to epoch
            %       condition:           optional string specifying the name of the condition,
            %                            default: unknown
            %       subjectID:           unique identidier of the epoched data, obj.uuid or a 
            %                            combination of the former and obj.sessionUUID is recommended,
            %                            default: obj.sessionUUID
            %       preStimulusLatency:  two elements vector specifying the segment in the epoch 
            %                            considered pre-stimulus (or base line), default: [timeLimits(1) 0]
            %
            %
            % Output argument:
            %       epochObject:         handle to the epoched object
            %
            % Usage:
            %       eegObj     = mobilab.allStreams.item{ eegItem };
            %       eventType  = {'eventLabel_x'}; % for more than one type use: {'eventLabel_x', 'eventLabel_y', ...} 
            %       timeLimits = [-2 2];           % two seconds before and after the event (it does not have to be a symmetric window)
            %       channels   = 85;               % channel to epoch (could be more than one). Observe that in case of ICA data,
            %                                      % channels correspond to independent components activation.
            %       condition  = 'target1';        
            %       preStimulusLatency = [-1.7 -1.2]; % base-line: between 1.7 and 1.2 seconds before the stimulus/response
            %
            %       epObj = eegObj.epoching( eventType, timeLimits, channels, condition, eegObj.sessionUUID, preStimulusLatency);
            %       plot(epObj);
            %       [~,ersp,itc,frequency,time] = epObj.waveletTimeFrequencyAnalysis;
            %      
            % See eegEpoch for more details

            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 5, condition = 'unknown';end 
            if nargin < 6, subjectID = obj.uuid;end
            if nargin < 7, warning('MoBILAB:noChannels','Undefined preStimulusLatency, assuming half of the epoch.'); preStimulusLatency = [];end
            
            [data,time,eventInterval] = epoching@coreStreamObject(obj,eventLabelOrLatency, timeLimits, channels);
            
            if isempty(preStimulusLatency)
                preStimulusLatency = [timeLimits(1) 0];
            elseif any(preStimulusLatency < timeLimits(1)) || any(preStimulusLatency > timeLimits(end))
                preStimulusLatency = [timeLimits(1) 0];
            elseif length(preStimulusLatency) ~= 2
                error('preStimulusLatency must be a two elements vector.');
            end
            [~,loc1] = min(abs(time - preStimulusLatency(1)));
            [~,loc2] = min(abs(time - preStimulusLatency(2)));
            preStimulusLatency = [loc1 loc2];
            
            epochObj = eegEpoch(data,time,obj.label(channels),condition,eventInterval,subjectID,preStimulusLatency);
        end
        %%
        function epochObj = epochingTW(obj,latency, channels, condition,subjectID)
            % Creates epoch objects. Epoch objects don't result in new additions to the tree. 
            % Even when an epoch object manages its data through a memory mapped file they are
            % regarded as temporal variables, once the object is destroyed its binary file is
            % deleted. TW stands for time warping. This method makes trials between two set of
            % events/latencies, because each trial can have a slightly different size the resulting
            % trials are interpolated to a common time axis.
            % 
            % Input argument: 
            %       latency:     two columns matrix with the set of start end latencies (in seconds)
            %                    for each trial.
            %       channels:    indices of the channels to epoch
            %       condition:   optional string specifying the name of the condition, default: unknown
            %       subjectID:   unique identidier of the epoched data, obj.uuid or a combination of the
            %                    former and obj.sessionUUID is recommended, default: obj.sessionUUID
            %
            % Output arguments:
            %       epochObject: handle to the epoched object
            %
            % Usage:
            %       eegObj    = mobilab.allStreams.item{ eegItem };
            %       latency_walk = eegObj.event.getLatencyForEventLabel('701'); % Go event marker
            %       latency_stop = eegObj.event.getLatencyForEventLabel('702'); % Stop event marker
            %       latency = [latency_walk(:) latency_stop(:)];
            %       channels  = 85;             % channel to epoch (could be more than one). Observe that in case of ICA data,
            %                                   % channels correspond to independent components activation.
            %       condition  = 'walking';    
            %
            %       epObj      = eegObj.epochingTW(latency, channels, condition, eegObj.sessionUUID);
            %       plot(epObj);
            %       [~,ersp,itc,frequency,time] = epObj.waveletTimeFrequencyAnalysis;
            %      
            % See eegEpoch for more details
            
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 4, condition = 'unknown';end
            if nargin < 5, subjectID = obj.uuid;end 
            
            [data,time,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
            epochObj = eegEpoch(data,time,obj.label(channels),condition,eventInterval,subjectID);
        end
        
    end
    methods(Hidden = true)
        function loadElectrodeWizard(obj,file)
            warning('This method will be deprecated, instead use ''readMontage'' with the same input arguments.');
            readMontage(obj,file);
        end
        %%
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid  = obj.uuid;
            end
            newHeader = createHeader@dataStream(obj,commandHistory);
            if ~isempty(newHeader), return;end
            
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            uuid = generateUUID;
            metadata.uuid = uuid;
            path = fileparts(obj.binFile);
            
            switch commandHistory.commandName
                case 'reReference'
                    prename = 'ref_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    channels2BeReferenced = commandHistory.varargin{1};
                    ind = ismember(obj.label,channels2BeReferenced);
                    metadata.label = obj.label(ind);
                    metadata.numberOfChannels = length(metadata.label);
                    if ~isempty(obj.channelSpace)
                        metadata.channelSpace = metadata.channelSpace(ind,:);
                    end
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                        
                otherwise
                    error('Cannot make a copy of this object. Please provide a valid ''command history'' instruction.');
            end
            newHeader = metadata2headerFile(metadata);
        end
        %%
        function disp(obj)
            string = sprintf('  channelSpace:         <%ix3 double>\n',size(obj.channelSpace,1));
            disp@coreStreamObject(obj)
            fprintf(string);
        end
        %%
        function properyArray = getPropertyGridField(obj)
            dim  = size(obj.channelSpace,1);
            properyArray = getPropertyGridField@coreStreamObject(obj);
            properyArray{1}{end+1} = 'channelSpace';
            properyArray{2}{end+1} = ['<' num2str(dim(1)) 'x3 ' obj.precision '>'];
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Add sensor locations');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'readMontage',-1});
            jmenu.add(menuItem);
            %--
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Re-reference');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'reReference',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'filter',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot spectrum');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrum',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Inspect');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'inspect',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Export to EEGLAB');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'EEGstructure',0});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Annotation');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Generate batch script');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@generateBatch_Callback,obj});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="maroon">Delete object</HTML>');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});
            jmenu.add(menuItem);
        end
    end
    methods(Static), function triggerSaveHeader(~,evnt), saveHeader(evnt.AffectedObject, 'f'); end;end
end

%--
function copyMMF(EEG,streamObj)
% bytes = dir(streamObj.binFile);
% bytes = bytes.bytes/1e9;
% if bytes < 0.5
fid = fopen(EEG.data,'w');
if fid < 0, return;end
if ~iscell(streamObj), streamObj = {streamObj};end
N = length(streamObj{1}.timeStamp);
N2 = length(streamObj);
% precision = streamObj{1}.precision;
precision = 'single';
for it=1:N2, if ~strcmp(streamObj{it}.isMemoryMappingActive,'active'), fclose(fid);return;end;end
bufferSize = 1000;
streamObj{1}.container.container.initStatusbar(1,N,'Creating EEG.data...');
for it=1:bufferSize:N
    try writeThis = [];
        for jt=1:N2, writeThis = [writeThis;streamObj{jt}.data(it:it+bufferSize-1,:)'];end%#ok        
    catch writeThis = [];%#ok
        for jt=1:N2, writeThis = [writeThis;streamObj{jt}.data(it:end,:)'];end%#ok
    end
    fwrite(fid,writeThis(:),precision);
    streamObj{1}.container.container.statusbar(it);
end
streamObj{1}.container.container.statusbar(inf);
fclose(fid);
end
