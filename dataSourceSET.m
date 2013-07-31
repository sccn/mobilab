% Definition of the class dataSourceSET. This class imports EEGLAB set files into MoBILAB.
% If the set file contains an ICA decomposition it creates two objects, one for EEG and the
% other one (its child) for the ICA data.
% 
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Oct-2012

classdef dataSourceSET < dataSource
    methods
        function obj = dataSourceSET(varargin)
            % Creates a dataSourceSET object.
            % 
            % Input arguments:
            %       file:              EEGLAB set file
            %       mobiDataDirectory: path to the directory where the collection of
            %                          files will be stored
            %                          
            % Output arguments:        
            %       obj:               dataSource object (handle)
            %
            % Usage:
            %        % file: EEGLAB set file
            %        % mobiDataDirectory: path to the directory where the collection of
            %        %                    files are or will be stored.
            %
            %        obj = dataSourceSET( file,  mobiDataDirectory);

            if nargin==1, varargin = varargin{1};end
            if length(varargin) < 2, error('Not enough input arguments.');end
            sourceFileName = varargin{1};
            mobiDataDirectory = varargin{2};
            [path,~,ext] = fileparts(sourceFileName);
            if ~strcmpi(ext,'.set'), error('prog:input',['dataSourceSET cannot read ''' ext ''' format.']);end
            if strcmp(path,mobiDataDirectory), error('Source and destiny folders must be different.');end
            
            uuid = generateUUID;
            obj@dataSource(mobiDataDirectory,uuid);
            obj.checkThisFolder(mobiDataDirectory);
            
            obj.container.lockGui;
            EEG = pop_loadset(sourceFileName);
            dim = size(EEG.data);
            dim(end+1) = 1;
            uuid = generateUUID;
            binFile = [obj.mobiDataDirectory filesep EEG.setname '_' uuid '_' obj.sessionUUID '.bin'];
            header  = [obj.mobiDataDirectory filesep EEG.setname '_' uuid '_' obj.sessionUUID '.hdr'];
            
            eventObj = event;
            eventObj = eventObj.addEvent(cell2mat({EEG.event.latency}),{EEG.event.type});
            fiducials = [];
            if ~isempty(EEG.chanlocs)
                labels = {EEG.chanlocs.labels};
                try
                    [elec,lab] = readMontage(EEG.chaninfo.filename);
                    I = ismember(lab,labels);
                    channelSpace = elec(I,:);
                    I = ismember(lab,'fidnz') | ismember(lab,'nasion');
                    if any(I), fiducials.nasion = elec(I,:);end
                    
                    I = ismember(lab,'fidt9') | ismember(lab,'lpa');
                    if any(I), fiducials.lpa = elec(I,:);end
                    
                    I = ismember(lab,'fidt10') | ismember(lab,'rpa');
                    if any(I), fiducials.rpa = elec(I,:);end
                    
                    I = ismember(lab,'fidt11') | ismember(lab,'vertex');
                    if any(I), fiducials.vertex = elec(I,:);end
                    
                catch %#ok
                    channelSpace = [cell2mat({EEG.chanlocs.X})' cell2mat({EEG.chanlocs.Y})' cell2mat({EEG.chanlocs.Z})'];
                end
            else
                labels = cell(EEG.nbchan,1);
                channelSpace = nan(EEG.nbchan,3);
                for it=1:EEG.nbchan, labels{it} = num2str(it);end
            end
            if length(EEG.times) == prod(dim(2:3))
                timeStamp = EEG.times;
            else timeStamp = (0:prod(dim(2:3))-1)/EEG.srate;
            end
            parentCommand.commandName = 'dataSourceSET';
            parentCommand.varargin{1} = sourceFileName;
            parentCommand.varargin{2} = mobiDataDirectory;
            
            metadata = struct('binFile',binFile,'header',header,'name',EEG.setname,'timeStamp',[],'numberOfChannels',EEG.nbchan,'precision','double',...
                'uuid',uuid,'sessionUUID',obj.sessionUUID,'writable',true,'unit','none','owner',struct('name',obj.container.preferences.username,...
                'organization',obj.container.preferences.organization,'email',obj.container.preferences.email),'hardwareMetaData',[],...
                'parentCommand',parentCommand,'label',[],'event',eventObj.saveobj,'notes','','artifactMask',sparse(length(timeStamp),...
                EEG.nbchan),'samplingRate',EEG.srate,'channelSpace',[],'class','eeg','fiducials',fiducials);
            metadata.timeStamp = timeStamp;
            metadata.label = labels;
            metadata.channelSpace = channelSpace;
            
            fid = fopen(binFile,'w');
            data = reshape(EEG.data,[dim(1) prod(dim(2:3))]);
            if isa(data,'mmo'), data = data(:,:);end
            fwrite(fid,data','double');
            fclose(fid);
            header = metadata2headerFile(metadata);
            obj.addItem(header);
            
            if ~isempty(EEG.icawinv) && ~isempty(EEG.icaweights) && ~isempty(EEG.icasphere)
                uuid = generateUUID;
                metadata.uuid = uuid;
                metadata.name = ['ica_' metadata.name];
                metadata.class = 'icaEEG';
                metadata.numberOfChannels = length(EEG.icachansind);
                metadata.label = cell(metadata.numberOfChannels,1);
                metadata.channelSpace = metadata.channelSpace(EEG.icachansind,:);
                metadata.icawinv    = EEG.icawinv;
                metadata.icaweights = EEG.icaweights;
                metadata.icasphere  = EEG.icasphere;
                
                for it=1:metadata.numberOfChannels, metadata.label{it} = ['IC' num2str(it)];end
                metadata.binFile = [obj.mobiDataDirectory filesep 'ica_' EEG.setname '_' uuid '_' obj.sessionUUID '.bin'];
                fid = fopen(metadata.binFile,'w');
                data = reshape(EEG.icaact,[dim(1) prod(dim(2:3))]);
                if isa(data,'mmo'), data = data(:,:);end
                fwrite(fid,data','double');
                fclose(fid);
                
                header = metadata2headerFile(metadata);
                obj.addItem(header);
            end
            
            obj.connect;
            obj.updateLogicalStructure;
            obj.container.lockGui;
        end
    end
end