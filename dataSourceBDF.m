%% dataSourceBDF class
%
%
%
%% Properties:
% - This class does not define new properties. -
%% Methods:
% - This class does not implement new methods. -
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 05-Apr-2011

%%
classdef dataSourceBDF < dataSource
    methods
        %%
        function obj = dataSourceBDF(sourceFileName,mobiDataDirectory,configList)
            checkNumberInputArguments(2, 3);
            if nargin < 3, configList = [];else configList{end+1,1}  = []; end
            [~,~,ext] = fileparts(sourceFileName);
            if ~strcmp(ext,'.bdf'), error('prog:input',['dataSourceBDF cannot read ''' ext ''' format.']);end
            tmp = what(mobiDataDirectory);
            if isempty(tmp), error('prog:input','Second argument must be a valid folder.');end
            mobiDataDirectory = tmp.path;
            for it=1:size(configList,1), if isempty(configList{it,1}), break;end;end
            configList(it:end,:) = [];
            
            DAT = openbdf(sourceFileName);
            [DAT,S] = readbdf(DAT,1:DAT.Head.NRec);
            DAT.Record = DAT.Record.';
            precision = class(DAT.Record);
            if size(DAT.Record,2)-2 ~= sum(cell2mat(configList(:,2))), error('MoBILAB:wrongConfigList',['The configuration list does not match the number of channels in the files.']);end
            
            uuid = generateUUID;
            obj@dataSource(mobiDataDirectory,uuid);
            obj.checkThisFolder(mobiDataDirectory);
            
            % Show time!!!
            parentCommand.commandName = 'dataSourceBDF';
            parentCommand.varargin{1} = sourceFileName;
            parentCommand.varargin{2} = mobiDataDirectory;
            parentCommand.varargin{3} = configList;
            
            SampleRate = DAT.Head.SampleRate(1);
            numberOfChannels = DAT.Head.NS;
            event_ch = find(ismember(DAT.Head.Label,['EventCode' repmat(' ',1,size(DAT.Head.Label,2)-length('EventCode'))],'rows'),1);
            if isempty(event_ch), event_ch = numberOfChannels;end
            
            stream_count = 1;
            index = cell(1);
            name_list = [];
            for it=1:size(configList,1)
                if ~isempty(configList{it,1})
                    name_list{stream_count} = lower(configList{it,1});%#ok
                    stream_count = stream_count+1;
                end
            end
            hardwareMetaDataObj = hardwareMetaData;

            if ~isempty(event_ch)
                eventObj = event(DAT.Record(:,event_ch));
            else eventObj = event;
            end
            timeStamp = (0:size(DAT.Record,1)-1)/SampleRate;
            
            pointer1 = 1;
            Nstreams = size(configList,1);
            for stream_count=1:Nstreams

                Nch = configList{stream_count,2};
                if isnan(Nch), error('prog:input','Invalid Number of Channels. Check the configuration.');end
                pointer2 = pointer1+Nch-1;
                index{stream_count} = pointer1:pointer2;
                if pointer2 > numberOfChannels, error('prog:input','Invalid Number of Channels. Check the configuration.');end
                labels = cell(Nch,1);
                Labels = DAT.Head.Label(index{stream_count},:);
                for it=1:Nch, labels{it} = deblank(Labels(it,:));end
                name{stream_count} = lower(configList{stream_count,1});%#ok
                ind = ismember(name_list, name{stream_count});
                if sum(ind)>1
                    name{stream_count} = [name{stream_count} num2str(sum(ind))];%#ok 
                    name_list{end+1} = name{stream_count};%#ok
                end
                
                uuid = generateUUID;
                binFile = [obj.mobiDataDirectory filesep name{stream_count} '_' uuid '_' obj.sessionUUID '.bin'];
                header  = [obj.mobiDataDirectory filesep name{stream_count} '_' uuid '_' obj.sessionUUID '.hdr'];
                fid2 = fopen(binFile,'w');
                fwrite(fid2,DAT.Record(:,index{stream_count}),precision);
                fclose(fid2);
                hardwareMetaDataObj.name = name{stream_count};
                
                metadata.binFile = binFile;
                metadata.header = header;
                metadata.dob = now;
                metadata.name = name{stream_count};
                metadata.timeStamp = timeStamp;
                metadata.samplingRate = SampleRate;
                metadata.numberOfChannels = Nch;
                metadata.label = labels;
                metadata.writable = false;
                metadata.parentCommand = parentCommand;
                metadata.hardwareMetaData = hardwareMetaDataObj;
                metadata.precision = precision;
                metadata.event.label = eventObj.label;
                metadata.event.hedTag = eventObj.hedTag;
                metadata.event.latencyInFrame = eventObj.latencyInFrame;
                metadata.uuid = uuid;
                metadata.sessionUUID = obj.sessionUUID;
                metadata.unit = 'none';
                switch name{stream_count}
                    case {'phasespace' 'optitrack'}
                        metadata.class = 'mocap';
                    case 'wii', metadata.class = 'wii';
                    otherwise
                        metadata.auxChannel.label = {};
                        metadata.auxChannel.data = [];
                        metadata.class = 'eeg';
                end
                header = metadata2headerFile(metadata);
                obj.addItem(header);
                clear metadata;
                pointer1 = pointer2+1;
            end
            
            obj.connect;
            obj.expandAroundBoundaryEvents;
            obj.findSpaceBoundary;
            obj.updateLogicalStructure;
            obj.save(obj.mobiDataDirectory);
        end
    end
end