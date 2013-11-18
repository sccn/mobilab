classdef basicSegment
    properties
        segmentName
        startLatency
        endLatency
        uuid = '';
    end
    properties(Hidden)
        parentCommand
    end
    properties(Dependent)
        history
    end
    methods
        %%
        function obj = basicSegment(varargin)
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            N = length(varargin);
            if isnumeric(varargin{1})
                obj.startLatency = varargin{1}(:,1)';
                obj.endLatency   = varargin{1}(:,2)';
                if N < 2
                    obj.segmentName = 'unnamed';
                elseif ischar(varargin{2}), 
                    obj.segmentName = varargin{2};
                else
                    obj.segmentName = 'unnamed';
                end    
                obj.parentCommand.varargin{1} = 'start';
                obj.parentCommand.varargin{2} = 'end';
                return
            end
            
            if N < 3, error('Not enough input arguments.');end
            if ~isa(varargin{1},'coreStreamObject'), error('First argument must be an object.');end
            dataStreamObj = varargin{1};
            obj.parentCommand.uuid = dataStreamObj.uuid;
            
            if ~iscellstr(varargin{2}), error('''startEventLabel'' must be a cell of strings.');end
            startEventLabel = varargin{2};
            
            if ~iscellstr(varargin{3}), error('''endEventLabel'' must be a cell of strings.');end
            endEventLabel = varargin{3};
            
            if N < 4, segmentName = [startEventLabel{1} '_' endEventLabel{1}];else segmentName = varargin{4};end %#ok
            if N < 5, checkSegmentLength = true;else checkSegmentLength = varargin{5};end
            
            obj.segmentName = segmentName;%#ok
            obj.startLatency = [];
            obj.endLatency = [];
              
            endPointer = [];
            Nstart = length(startEventLabel);
            Nend = length(endEventLabel);
            for it=1:Nend
                endPointer = [endPointer dataStreamObj.event.getLatencyForEventLabel(endEventLabel{it})];%#ok
            end
            ptrTable.startlabel = {};
            ptrTable.endlabel = {};
            One = ones(length(endPointer),1);
            for it=1:Nstart
                startPointer = dataStreamObj.event.getLatencyForEventLabel(startEventLabel{it});
                Nstart = length(startPointer);
                if ~isempty(startPointer)
                    M = endPointer'*ones(1,Nstart) - One*startPointer;
                    indStart = 1:Nstart;
                    indEnd = zeros(1,Nstart);
                    for kk=indStart
                        nzI = find(M(:,kk)>0);
                        if ~isempty(nzI)
                            [~,ind] = min(M(nzI,kk));
                            indEnd(kk) = nzI(ind);
                        end
                    end
                    Iz = indEnd==0;
                    indStart(Iz) = [];
                    indEnd(Iz) = [];
                    
                    if ~isempty(indStart)
                        repInd = find(diff(indEnd)==0);
                        if ~isempty(repInd)
                            repInd = [repInd repInd(end)+1]; %#ok
                            [~,keepThis] = min(endPointer(indEnd(repInd))-startPointer(repInd));
                            rmThis = setdiff(repInd,repInd(keepThis));
                            indStart(rmThis) = [];
                            indEnd(rmThis) = [];
                        end
                    end
                    if ~isempty(indStart)
                        obj.startLatency = [obj.startLatency dataStreamObj.timeStamp(startPointer(indStart))];
                        obj.endLatency   = [obj.endLatency dataStreamObj.timeStamp(endPointer(indEnd))];
                        ptrTable.startlabel = cat(2,ptrTable.startlabel,repmat(startEventLabel(it),1,numel(startPointer(indStart))));
                    end
                end
            end
            if isempty(obj.endLatency) || isempty(obj.startLatency), obj.startLatency = -1;return;end
            if checkSegmentLength && length(obj.startLatency) > 3
                x = obj.endLatency-obj.startLatency;
                stats = bootstrp(100,@(x)[median(x) std(x)],x);
                stats = mean(stats);
                rmIndex = x < stats(1) - 2*stats(2) | x > stats(1) + 2*stats(2);
                obj.startLatency(rmIndex) = [];
                obj.endLatency(rmIndex) = [];
                ptrTable.startlabel(rmIndex) = [];
            end
            ptrTable.endlabel = repmat(endEventLabel(1),1,numel(obj.endLatency));
            obj.parentCommand.objectIndex = num2str(dataStreamObj.container.findItem(dataStreamObj.uuid));
            obj.parentCommand.varargin{1} = startEventLabel;
            obj.parentCommand.varargin{2} = endEventLabel;
            obj.parentCommand.varargin{3} = segmentName;%#ok
            obj.parentCommand.ptrTable = ptrTable;
            if ismac, [~,hash] = system('uuidgen'); else hash = java.util.UUID.randomUUID;end
            obj.uuid = char(hash);
        end
        %%
        function history = get.history(obj)
            if isfield(obj.parentCommand,'objectIndex')
                A = evalin('base','whos');
                dSname = [];
                for it=1:length(A)
                    if strcmp(A(it).name,'allDataStreams')
                        dSname = A(it).name;
                        break
                    end
                end
                if isempty(dSname), dSname = 'mobilab.allStreams';end
                history = ['basicSegment(' dSname '.item{' obj.parentCommand.objectIndex '},'];
                for it=1:length(obj.parentCommand.varargin)-1, history = [history 'varargin{' num2str(it) '},'];end %#ok
                history = [history 'varargin{' num2str(it+1) '});'];
                assignin('caller','varargin',obj.parentCommand.varargin);
            else
                tmp = [];
                for it=1:length(obj.startLatency), tmp = [tmp ' ' num2str(obj.startLatency(it)) ' ' num2str(obj.endLatency(it)) ';' ];end %#ok
                tmp(end) = [];
                history = ['segObj=basicSegment([' tmp '],''' obj.segmentName ''');'];
            end
        end
        %%
        function flag = eq(obj,obj2)
            flag = strcmp(obj.segmentName,obj2.segmentName);
            flag = flag & length(obj.startLatency) == length(obj2.startLatency);
            flag = flag & length(obj.endLatency) == length(obj2.endLatency);
            if ~flag, return;end
            flag = flag & all(obj.startLatency == obj2.startLatency);
            flag = flag & all(obj.endLatency == obj2.endLatency);
        end
        %%
        function segObj = apply(obj,streamObj,channels)
            if nargin < 2, error('You must enter the stream you want to segment.');end
            if nargin < 3, channels = 1:streamObj.numberOfChannels;end
            header = obj.prepareArgumentsBeforeApply(streamObj,channels);
            segObj = streamObj.container.addItem(header);
        end
        %%
        function cobj = cat(obj,obj2)
            sLatency = [obj.startLatency obj2.startLatency];
            eLatency = [obj.endLatency obj2.endLatency];
            
            N = length(sLatency);
            time = linspace(min(sLatency),max(eLatency),25000)';
            I = false(length(time),1);
            for it=1:N, I = I | time >= sLatency(it) & time <= eLatency(it);end
            dI = [1;diff(I)];
            dI(end) = -1;
            %figure;plot(dI);
            
            pointer = time(dI==1);
            [~,loc] = min(abs(pointer*ones(1,N) - ones(length(pointer),1)*sLatency),[],2);
            sLatency = sLatency(loc);
            
            pointer = time(dI==-1);
            [~,loc] = min(abs(pointer*ones(1,N) - ones(length(pointer),1)*eLatency),[],2);
            eLatency = eLatency(loc);
            
            if strcmp(obj.segmentName,obj2.segmentName), sName = obj.segmentName;
            else sName = [obj.segmentName '+' obj2.segmentName];
            end
            cobj = basicSegment([sLatency' eLatency'],sName);
        end
        %%
        function writeDRF(obj,streamObj,drfFilename)
            if nargin < 3, error('Not enough input arguments.');end
            
            if length(streamObj) == 1 && ~iscell(streamObj), streamObj = {streamObj};end
            
            [drfFolder,drfName,ext] = fileparts(drfFilename); %#ok
            if isempty(drfFolder), drfFolder = pwd;end
            ext =  '.drf';
            Nstreams = length(streamObj);
            
            docNode = com.mathworks.xml.XMLUtils.createDocument('streamsample');
            streamsample = docNode.getDocumentElement;
            reservedItem = docNode.createElement('reserved');
            reservedItem.setAttribute('bytes',num2str(streamObj{1}.hardwareMetaData.reserved_bytes));
            streamsample.appendChild(reservedItem);
            
            filesampleItem = docNode.createElement('filesample');
            stream_count = docNode.createElement('stream_count');
            stream_count.setAttribute('bytes','4');
            filesampleItem.appendChild(stream_count);
            sample = docNode.createElement('sample');
            
            offset = zeros(streamObj{1}.hardwareMetaData.offset,1,'int8');
            offset(1) = Nstreams;
            
            % Uff this is so boring!! 
            for streamsIt=1:Nstreams
                if isempty(streamObj{streamsIt}.hardwareMetaData) 
                    error('''hardwareMetaData'' field is not available, you have to re-import the .drf file in order to get this info.'); 
                end
                stream = streamObj{streamsIt}.hardwareMetaData.insertInHeader(docNode);
                sample.appendChild(stream);
            end
            filesampleItem.appendChild(sample);
            streamsample.appendChild(filesampleItem);
            
            xmlwrite([drfFolder filesep drfName '.xml'],docNode);
            
            eventChannel = cell(length(streamObj),1);
            uV = cell(length(streamObj),1);
            for streamsIt=1:Nstreams
                hardwareMetaDataObj = streamObj{streamsIt}.hardwareMetaData;
                eventChannel{streamsIt} = streamObj{streamsIt}.event.event2vector(streamObj{streamsIt}.timeStamp);
                if strcmp(hardwareMetaDataObj.name,'wii')
                    uV{streamsIt} = hardwareMetaDataObj.uV(1);
                else
                    for kk=1:length(hardwareMetaDataObj.uV), uV{streamsIt} = [uV{streamsIt} repmat(hardwareMetaDataObj.uV(kk),1,hardwareMetaDataObj.count(kk))];end
                end
            end
            
            I = false(1,length(streamObj{1}.timeStamp));
            for it=1:size(obj.startLatency,1), I = I | streamObj{1}.timeStamp >= obj.startLatency & streamObj{1}.timeStamp <= obj.endLatency;end
            I = find(I);
            N = length(I);
            fid = fopen([drfFolder filesep drfName ext],'w');
            hwait = waitbar(0,'writing binary data...','Color',[0.66 0.76 1]);
            for it=1:N
                fwrite(fid,offset);
                for streamsIt=1:Nstreams
                    hardwareMetaDataObj = streamObj{streamsIt}.hardwareMetaData;
                    fwrite(fid,hardwareMetaDataObj.originalTimeStamp(I(it)),'int32');
                    fwrite(fid,eventChannel{streamsIt}(I(it)),'int32');
                    fwrite(fid,streamObj{streamsIt}.numberOfChannels,'int32');
                    fwrite(fid,hardwareMetaDataObj.item_size,'int32');
                    precision  = sprintf('int%i',8*hardwareMetaDataObj.item_size);
                    if strcmp(streamObj{streamsIt}.isMemoryMappingActive,'not active')
                         fwrite(fid,zeros(streamObj{streamsIt}.numberOfChannels,1,'single'),precision);
                    else fwrite(fid,streamObj{streamsIt}.data(I(it),:).*uV{streamsIt},precision);
                    end
                end
                waitbar(it/N,hwait);
            end
            close(hwait);
            fclose(fid);
        end
        %%
        function header = prepareArgumentsBeforeApply(obj,streamObj,channels)
            if nargin < 2, channels = 1:streamObj.numberOfChannels;end
            
            metadata = streamObj.saveobj;
            metadata.numberOfChannels = length(channels);
            metadata.name  = [obj.segmentName '_' streamObj.name];
            metadata.uuid  = generateUUID;
            metadata.label = streamObj.label(channels(:));
            
            path = fileparts(streamObj.binFile);
            metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' streamObj.sessionUUID '.bin']);
            metadata.header  = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' streamObj.sessionUUID '.hdr']);
            
            I = false(size(streamObj.timeStamp));
            for it=1:length(obj.startLatency), I = I | (streamObj.timeStamp >= obj.startLatency(it) & streamObj.timeStamp < obj.endLatency(it));end
            fid = fopen(metadata.binFile,'w');
            streamObj.container.container.initStatusbar(1,metadata.numberOfChannels,'Segmenting...');
            for it=1:metadata.numberOfChannels
                writeThis = streamObj.data(I,channels(it));
                fwrite(fid,writeThis(:),streamObj.precision);
                streamObj.container.container.statusbar(it);
            end
            fclose(fid);
            
            metadata.timeStamp = streamObj.timeStamp(I);
            metadata.artifactMask = streamObj.artifactMask(I,channels);
            metadata.writable = true;
            Ones1 = ones(1,length(obj.startLatency));
            Ones2 = ones(length(metadata.timeStamp),1);
            [~,locStart] = min(abs(metadata.timeStamp(:)*Ones1 - Ones2*obj.startLatency));
            [~,locEnd]   = min(abs(metadata.timeStamp(:)*Ones1 - Ones2*obj.endLatency));
            metadata.event = event;
            if isfield(obj.parentCommand,'ptrTable')
                metadata.event = metadata.event.addEvent(locStart,obj.parentCommand.ptrTable.startlabel);
                metadata.event = metadata.event.addEvent(locEnd,obj.parentCommand.ptrTable.endlabel);
            else
                metadata.event = metadata.event.addEvent(locStart,obj.parentCommand.varargin{1});
                metadata.event = metadata.event.addEvent(locEnd,obj.parentCommand.varargin{2});
            end
            if ~isempty(streamObj.event.latencyInFrame)
                latEvents = streamObj.timeStamp(streamObj.event.latencyInFrame);
                [~,loc1,loc2] = intersect(metadata.timeStamp,latEvents);
                labels = streamObj.event.label(loc2);
                metadata.event = metadata.event.addEvent(loc1,labels);
            end
                                    
            metadata.parentCommand.commandName = 'apply';
            metadata.parentCommand.uuid = streamObj.uuid;
            metadata.parentCommand.segmentName = obj.segmentName;
            metadata.parentCommand.channels = channels;
            if isfield(metadata.parentCommand,'varargin')
                metadata.parentCommand = rmfield(metadata.parentCommand,'varargin');
            end
            header = metadata2headerFile(metadata);
        end
    end
end
