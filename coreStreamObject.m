% Definition of the class coreStreamObject. This class serves as base to all
% stream objects in MoBILAB toolbox.
%
% For more details visit: https://code.google.com/p/mobilab/ 
%
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Apr-2011

classdef coreStreamObject < handle
    properties
        binFile   % Pointer to the file containing the data that is going to
                  % be memory mapped. It has the following structure:
                  %          path + object name + _ uuid _ sessionUUID .bin
        
        header    % Pointer to the file containing the metadata, it has the
                  % following structure:
                  %          path + object name + _ uuid _ sessionUUID .hdr
        notes
        mmfObj
    end
    properties(GetAccess = public, SetAccess = protected, AbortSet = true)
        name               % Name of the object. The name of an object 
                           % reflects the operation that has created it.
                           
        timeStamp          % Array of double precision numbers containing
                           % the time stamps of each sample in seconds.
                           
        numberOfChannels   % Number of channels.
        
        precision          % Data precision.
        
        uuid               % Universal unique identifier. Each object has 
                           % its own uuid.
                           
        sessionUUID        % UUID common to all the objects belonging to the
                           % same session. A session is defined as a raw file
                           % or collection of files recorded the same day and
                           % its derived processed objects.
                           
        writable           % If true the object can be modified and even deleted,
                           % otherwise is considered raw data and it cannot be modified.
                           
        unit               % Cell array of strings specifying the unit of each channel.
                           
        hardwareMetaData
        parentCommand
    end
    properties(GetAccess = public, SetAccess = public, AbortSet = true)
        samplingRate  % Sampling rate in hertz.
        
        label         % Cell array of strings specifying the label of each channel.
        
        event         % Event object that contain event markers relative to the time base of
                      % the object where it is contained.
                      
        container     % Pointer to the dataSource object.
                      
        auxChannel    % Auxiliary channels whose type may be not the same as the data contained in 
                      % obj.data. For instance a Trigger channel will go here. auxChannel.data contains
                      % the data and auxChannel.label the channel labels.
    end
    properties(Dependent)
        data      % Matrix size number of samples (same as time stamps) by numberOfChannels. 
                  % Data is a dependent property that access the time series stored in a binary
                  % file through a memory mapping file object. This makes possible working with
                  % data sets of any size, even if they don't fit in memory.
                  
        parent    % Pointer to the parent object. Object containing raw data have no parent, 
                  % in this case it returns empty.
                  
        children  % Cell array containing immediate descendant objects.
        
        history   % Command who has created the object. Calling this command will produce exactly
                  % the same data set. This field is very useful for the creation of scripts.
    end
    properties(GetAccess = public, SetAccess = private, Hidden=true)
        dob
    end
    properties(Constant)
        version = 3;
    end
    methods
        %%
        function obj = coreStreamObject(header)
            % Creates a coreStreamObject. As this is an abstract class it 
            % cannot be instantiated on its own. Therefore, this constructor
            % must be called only from within a child class constructor.
            % 
            % Input arguments:
            %       header: header file (string)
            %
            % Output arguments:
            %       obj: coreStreamObject (handle)
            % 
            % Usage: 
            %       obj@coreStreamObject(header);

            if nargin < 1, error('Not enough input parameters.');end
            obj.header = header;
            obj.binFile = [header(1:end-3) 'bin'];
            if ~exist(obj.binFile,'file'), error('The associated binary file %s is missing.',obj.binFile);end
            warning off                 %#ok
            load(header,'-mat','notes');
            if exist('notes','var') && ~isempty(notes), obj.notes = mobiAnnotator(obj,notes); else obj.notes = mobiAnnotator(obj);end %#ok
            load(header,'-mat','dob');
            warning on                  %#ok
            if exist('dob','var')
                obj.dob = dob;          %#ok
            else desc = dir(obj.binFile);
                dob = desc.datenum;     %#ok
                obj.dob = dob;          %#ok
                save(obj.header,'-mat','-append','dob');
            end
            obj.connect;
        end
        %%
        function cobj = copyobj(obj,commandHistory)
            % Creates a new object calling the appropriated constructor.
            % 
            % Input arguments:
            %       parentCommand: structure with the following fields: 
            %                      1) commandName (string), 2) uuid (string),
            %                      uuid of the parent object, and 3) varargin,
            %                      cell array of input arguments.
            % 
            % Output arguments:
            %       cobj: handle tho the new object
            %
            % Usage:
            %       eegObj  = mobilab.allStreams.item{ eegItem };
            %       eegObj2 = copyobj( eegObj );
            
            if nargin < 2, commandHistory = struct('commandName','copyobj','uuid',obj.uuid);end
            obj.container.container.lockGui('Creating new header and binary files. Please wait...');
            newHeader = createHeader(obj,commandHistory);
            if isempty(newHeader), error('Cannot create the new object. Please provide a valid ''command history'' instruction.');end
            cobj = obj.container.addItem(newHeader);
            obj.container.container.lockGui;
        end
        %%
        function name = get.name(obj)
            if isempty(obj.name), obj.name = retrieveProperty(obj,'name');end
            name = obj.name;
        end
        %%
        function timeStamp = get.timeStamp(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.timeStamp')), timeStamp = obj.timeStamp;return;end
            if isempty(obj.timeStamp), obj.timeStamp = retrieveProperty(obj,'timeStamp');end
            timeStamp = obj.timeStamp;
        end
        function set.timeStamp(obj,timeStamp)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.timeStamp')), obj.timeStamp = timeStamp;return;end
            obj.timeStamp = timeStamp;
            if any(strcmp({stack.name},'coreStreamObject.addSamples')), return;end
            saveProperty(obj,'timeStamp',timeStamp);
        end
        %%
        function numberOfChannels = get.numberOfChannels(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.numberOfChannels')), numberOfChannels = obj.numberOfChannels;return;end
            if isempty(obj.numberOfChannels), obj.numberOfChannels = retrieveProperty(obj,'numberOfChannels');end
            numberOfChannels = obj.numberOfChannels;
        end
        function set.numberOfChannels(obj,numberOfChannels)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.numberOfChannels')), obj.numberOfChannels = numberOfChannels;return;end
            obj.numberOfChannels = numberOfChannels;
            saveProperty(obj,'numberOfChannels',numberOfChannels);
        end
        %%
        function label = get.label(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.label')), label = obj.label;return;end
            if isempty(obj.label), obj.label = retrieveProperty(obj,'label');end
            label = obj.label;
        end
        function set.label(obj,label)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.label')), obj.label = label;return;end
            obj.label = label;
            saveProperty(obj,'label',label);
        end
        %%
        function writable = get.writable(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.writable')), writable = obj.writable;return;end
            if isempty(obj.writable), obj.writable = retrieveProperty(obj,'writable');end
            writable = obj.writable;
        end
        function set.writable(obj,writable)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.writable')), obj.writable = writable;return;end
            if ~isempty(obj.writable) && ~obj.writable, error('This is raw data, it must be preserved. Try copyobj to make a copy of it.');end
            obj.writable = writable;
            saveProperty(obj,'writable',writable);
        end
        %%
        function uuid = get.uuid(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.uuid')), uuid = obj.uuid;return;end
            if isempty(obj.uuid), obj.uuid = retrieveProperty(obj,'uuid');end
            uuid = obj.uuid;
        end
        function set.uuid(obj,uuid)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.uuid')), obj.uuid = uuid;return;end
            if ~ischar(uuid), uuid = char(uuid);end
            obj.uuid = uuid;
            saveProperty(obj,'uuid',uuid);
        end
        %%
        function sessionUUID = get.sessionUUID(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.sessionUUID')), sessionUUID = obj.sessionUUID;return;end
            if isempty(obj.sessionUUID), obj.sessionUUID = retrieveProperty(obj,'sessionUUID');end
            sessionUUID = obj.sessionUUID;
        end
        function set.sessionUUID(obj,sessionUUID)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.sessionUUID')), obj.sessionUUID = sessionUUID;return;end
            if ~ischar(sessionUUID), sessionUUID = char(sessionUUID);end
            obj.sessionUUID = sessionUUID;
            saveProperty(obj,'sessionUUID',sessionUUID);
        end
        %%
        function parentCommand = get.parentCommand(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.parentCommand')), parentCommand = obj.parentCommand;return;end
            if isempty(obj.parentCommand), obj.parentCommand = retrieveProperty(obj,'parentCommand');end
            parentCommand = obj.parentCommand;
        end
        function set.parentCommand(obj,parentCommand)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.parentCommand')), obj.parentCommand = parentCommand;return;end
            obj.parentCommand = parentCommand;
            saveProperty(obj,'parentCommand',parentCommand);
        end
        %%
        function precision = get.precision(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.precision')), precision = obj.precision;return;end
            if isempty(obj.precision), obj.precision = retrieveProperty(obj,'precision');end
            precision = obj.precision;
        end
        function set.precision(obj,precision)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.precision')), obj.precision = precision;return;end
            obj.precision = precision;
            saveProperty(obj,'precision',precision);
        end
        %%
        function unit = get.unit(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.unit')), unit = obj.unit;return;end
            if isempty(obj.unit)
                try obj.unit = retrieveProperty(obj,'unit');
                catch ME
                    if strcmp(ME.identifier,'MoBILAB:unknownProperty')
                        obj.unit = 'none';
                        saveProperty(obj,'unit',obj.unit);
                    else ME.rethrow;
                    end
                end
            end
            unit = obj.unit;
        end
        function set.unit(obj,unit)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.unit')), obj.unit = unit;return;end
            obj.unit = unit;
            saveProperty(obj,'unit',unit);
        end
        %%
        function hardwareMetaData = get.hardwareMetaData(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.hardwareMetaData')), hardwareMetaData = obj.hardwareMetaData;return;end
            if isempty(obj.hardwareMetaData), obj.hardwareMetaData = retrieveProperty(obj,'hardwareMetaData');end
            hardwareMetaData = obj.hardwareMetaData;
        end
        function set.hardwareMetaData(obj,hardwareMetaData)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.hardwareMetaData')), obj.hardwareMetaData = hardwareMetaData;return;end
            obj.hardwareMetaData = hardwareMetaData;
            saveProperty(obj,'hardwareMetaData',hardwareMetaData);
        end
        
        %%
        function eventObj = get.event(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.event')), eventObj = obj.event;return;end
            if isempty(obj.event)
                try obj.event = event(retrieveProperty(obj,'event')); %#ok
                catch ME
                    if strcmp(ME.identifier,'MoBILAB:unknownProperty')
                       obj.event = event; %#ok
                       saveProperty(obj,'event',obj.event);
                    else ME.rethrow;
                    end
                end
            end 
            eventObj = obj.event;
        end
        function set.event(obj,eventObj)
            if ~isa(eventObj,'event'), error('''event'' field must be an object from the class ''event''.');end
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.event')), obj.event = eventObj;return;end
            obj.event = eventObj;
            if any(strcmp({stack.name},'coreStreamObject.addSamples')), return;end
            event.hedTag = eventObj.hedTag; %#ok
            event.label = eventObj.label;   %#ok
            event.latencyInFrame = eventObj.latencyInFrame; %#ok
            disp(['Saving: event in: ' obj.header]);        %#ok
            save(obj.header,'-mat','-append','event');      %#ok
        end
        %%
        function samplingRate = get.samplingRate(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.samplingRate')), samplingRate = obj.samplingRate;return;end 
            if isempty(obj.samplingRate), obj.samplingRate = retrieveProperty(obj,'samplingRate');end
            samplingRate = obj.samplingRate;
        end
        function set.samplingRate(obj,newSamplingRate)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.samplingRate')), obj.samplingRate = newSamplingRate;return;end
            if isempty(obj.samplingRate), obj.samplingRate = newSamplingRate;return;end
            if newSamplingRate < 1, error('MoBILAB:noSRchanged','Sampling rate less than 1? Really?');end
            if obj.isMemoryMappingActive
                resample(obj,newSamplingRate,[],1);
            else
                obj.samplingRate = newSamplingRate;
                saveProperty(obj,'samplingRate',obj.samplingRate)
            end
        end
        %%
        function auxChannel = get.auxChannel(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.auxChannel'))
                auxChannel = obj.auxChannel;
                return;
            end
            if isempty(obj.auxChannel),
                try obj.auxChannel = retrieveProperty(obj,'auxChannel');
                catch
                    obj.auxChannel.label = {};
                    obj.auxChannel.data = [];
                    auxChannel = obj.auxChannel;
                    saveProperty(obj,'auxChannel',auxChannel);
                end
            end
            auxChannel = obj.auxChannel;
        end
        function set.auxChannel(obj,auxChannel)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.auxChannel'))
                obj.auxChannel = auxChannel;
                return;
            end
            obj.auxChannel = auxChannel;
            saveProperty(obj,'auxChannel',auxChannel)
        end
        %%
        function data = get.data(obj)
            data = [];
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.saveobj')), return;end
            if ~obj.isMemoryMappingActive, return;end
            try data = obj.mmfObj.Data.x;
                if isfield(obj.mmfObj.Data,'y'), data = data + 1j*obj.mmfObj.Data.y;end
            catch ME
                error('MoBILAB:errorReadingBinFile','%s\nCannot read the binary file.',ME.message);
            end
        end
        function set.data(obj,data)
            if ~obj.isMemoryMappingActive, warning('prog:input','The file is not mapped, cannot write data into it.');return;end
            if ~obj.writable, error('Cannot overwrite raw data. Use copyobj to create a copy.');end
            sizeData = size(data);
            sizeObj  = size(obj);
            if sizeData(1) ~= sizeObj(1), error('MoBILAB:noRMSamples','Cannot remove samples by hand.');end
            obj.mmfObj.Writable = obj.writable;
            isComplex = isfield(obj.mmfObj.Data,'y');
            if prod(sizeData(2:end)) ~= prod(sizeObj(2:end)) || ~isa(data,obj.precision)
                obj.numberOfChannels = prod(sizeData(2:end));
                obj.precision = class(data);    
                obj.mmfObj = [];
                fid = fopen(obj.binFile,'w');
                fwrite(fid,real(data(:)),obj.precision);
                if isComplex
                    fwrite(fid,imag(data(:)),obj.precision);
                    obj.mmfObj = memmapfile(obj.binFile,'Format',{obj.precision sizeData 'x';obj.precision sizeData 'y'},'Writable',obj.writable);
                else
                    obj.mmfObj = memmapfile(obj.binFile,'Format',{obj.precision sizeData 'x'},'Writable',true);
                end
                fclose(fid);
            else
                obj.mmfObj.Data.x = real(data);
                if isComplex, obj.mmfObj.Data.y = imag(data);end
            end
            obj.mmfObj.Writable = false;
        end
        %%
        function history = get.history(obj)
            warning off %#ok
            load(obj.header,'-mat','history')
            warning on %#ok
            if ~exist('history','var')
                history = serializeCommand(obj);
                obj.history = history;
            end
        end
        function set.history(obj,history), save(obj.header,'-mat','-append','history');end %#ok
        %%
        function parent = get.parent(obj)
            if ~isa(obj.container,'dataSource') || ~isfield(obj.parentCommand,'uuid')
                parent = [];
                return;
            end
            index = obj.container.findItem(obj.parentCommand.uuid);
            parent = obj.container.item{index};
        end
        function children = get.children(obj)
            children = [];
            if ~isa(obj.container,'dataSource'), return;end
            index = obj.container.findItem(obj.uuid);
            indices = obj.container.getDescendants(index);
            if isempty(indices), return;end
            children = obj.container.item(indices);
        end
        %%
        function dim = size(obj,d)
            % Returns the two-element row vector containing the number of 
            % rows and columns in obj.data, where prod(size(obj.data)) is 
            % equal to length(obj.timeStamp) * obj.numberOfChannels. In case
            % of reshaping the data size will return a vector with many 
            % elements as new dimensions.
            % 
            % Input arguments:
            %       dim:       dimension, make size returns the length of 
            %                  the specified dimension
            %  
            % Output argument:
            %       dimVector: vector with the length of each dimension of
            %                  the data
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       dim    = size( eegObj );
            %       dim    = size( eegObj, 1); % number of time points
            
            if obj.isMemoryMappingActive, dim = obj.mmfObj.Format{2}; else dim = [0 0];end
            if nargin > 1, try dim = dim(d); catch dim = [];end;end %#ok
        end
        %%
        function reshape(obj,dim)
            % Reshape obj.data.
            % 
            % Input argument:
            %       dim:      vector with the length of each new dimension
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       dim      = size( mocapObj );
            %       reshape( mocapObj, [dim(1), 3 dim(2)/3]); 
            %       % reshapes the motion data in number of time points by 
            %       xyz values by number of position sensors
            
            checkNumberInputArguments(2, 2)
            if ~obj.isMemoryMappingActive, error('prog:input','Cannot access the memmapfile. Try ''obj.connetc;'' first.');end
            if prod(dim) ~= prod(obj.size), error('prog:input','To RESHAPE the number of elements must not change.');end
            if isfield(obj.mmfObj.Data,'y')
                 obj.mmfObj.Format = {obj.precision dim 'x';obj.precision dim 'y'};
            else obj.mmfObj.Format = {obj.precision dim 'x'};
            end
        end
        %%
        function delete(obj)
            if sum(size(obj)) == 0, return;end
            try delete(obj.notes);
                obj.disconnect; 
            catch ME
                warning('MoBILAB:objectDeleted',ME.message);
            end
        end
        %%
        function [t1,t2] = getTimeIndex(obj,timeValues)
            % Returns the indices correspondent to the input vector of time
            % stamps.
            % 
            % Input arguments:
            %       latencyInSeconds: vector of time stamps in seconds
            %                             
            % Output argument:          
            %       latencyInSamples: latency in samples correspondent to 
            %                         the time stamps.
            %
            % Usage:
            %       eegObj           = mobilab.allStreams.item{ eegItem };
            %       latencyInSeconds = [1.23 2 3.45] ); % time stamps in seconds
            %       latencyInSamples = eegObj.getTimeIndex( latencyInSeconds );
            %       samples          = eegObj.data( indices, :);

            if nargin < 2, timeValues = [-inf inf];end
            if ~isnumeric(timeValues), error('Input argument must be a vector of timestamps (in sec).');end
            index = interp1(obj.timeStamp,1:length(obj.timeStamp),timeValues(:)','nearest','extrap');
            if nargout==2
                t1 = index(1);
                t2 = index(2);
            else
                t1 = index;
            end
        end
        %%
        function inspect(obj,~)
            % Pops up a figure showing all the published properties of the 
            % object.
            properyArray = getPropertyGridField(obj);
            inputdlg(properyArray{1},obj.name,1,properyArray{2});
        end
        function val = isMemoryMappingActive(obj)
            % Returns true if the connection with the binary file is valid,
            % otherwise returns false.
            
            val = true;
            if isempty(obj.mmfObj), val = false;end;
        end
        %%
        function browserObj = dataStreamBrowser(obj,defaults)
            % Pops up a time series browser.
            browserObj = DataStreamBrowser(obj);
        end
        %%
        function jsonObj = serialize(obj)
            % Returns a serialized version of the object in a form of a 
            % JSON string.
            
            metadata = saveobj(obj);
            metadata.class = class(obj);
            metadata.size = size(obj);
            metadata.event = obj.event.uniqueLabel;
            metadata.writable = double(metadata.writable);
            metadata.history = obj.history;
            if isfield(metadata,'segmentUUID'), metadata.segmentUUID = char(metadata.segmentUUID);end
            if isfield(metadata,'originalStreamObj'), metadata.originalStreamObj = char(metadata.originalStreamObj);end
            if isfield(metadata,'animationParameters')
                metadata.hasStickFigure = 'no';
                if ~isempty(metadata.animationParameters.conn), metadata.hasStickFigure = 'yes';end
                metadata = rmfield(metadata,'animationParameters');
            end
            metadata = rmfield(metadata,{'parentCommand' 'timeStamp','hardwareMetaData'});
            jsonObj = savejson('',metadata,'ForceRootName', false);
        end
        %%
        function [data,time,eventInterval] = epoching(obj,eventLabelOrLatency, timeLimits, channels)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if iscellstr(eventLabelOrLatency)
                latency = obj.event.getLatencyForEventLabel(eventLabelOrLatency);
            elseif ischar(eventLabelOrLatency)
                latency = obj.event.getLatencyForEventLabel(eventLabelOrLatency);
            elseif isvector(eventLabelOrLatency)
                latency = eventLabelOrLatency;
            else error('First argument has to be a cell array with the event labels or a vector with of latencies (in samples).');
            end
            
            Nt = length(latency);
            if Nt < 1, error('None epoch could be extracted.');end
            t1 = timeLimits(1):1/obj.samplingRate:0;
            t2 = 1/obj.samplingRate:1/obj.samplingRate:timeLimits(2);
            time = [t1 t2];
            d1 = length(t1)-1;
            d2 = length(t2);
            data(length(time),Nt,length(channels)) = 0;
            rmThis = zeros(Nt,1);
            for k=1:Nt
                try data(:,k,:) = obj.data([latency(k)-d1:latency(k) latency(k)+1:latency(k)+d2],channels);
                catch rmThis(k) = k; %#ok
                end
            end
            rmThis(rmThis==0) = [];
            if ~isempty(rmThis), data(:,rmThis,:) = [];end
            eventInterval = diff(latency)/obj.samplingRate;
        end
        %%
        function [data,time,eventInterval] = epochingTW(obj,latency, channels)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            
            dim = size(latency);
            dl = round(mean(diff(latency,[],2)));
            Data = obj.mmfObj.Data.x;
            data = [];
            for it=1:dim(1)
                X = [];
                for jt=1:dim(2)-1 
                    ind = latency(it,jt):latency(it,jt+1);
                    y = Data(ind,channels);
                    x = linspace(1,dl(jt),length(ind))';
                    xi = (1:dl(jt))';
                    yi = interp1(x,y,xi,'linear');
                    X = cat(1,X,yi);
                end
                data = cat(2,data,X);
            end
            n = size(X,1)/2;
            time = linspace( -n, n, size(X,1))/obj.samplingRate;
            data = reshape(data, [length(time) length(channels) dim(1)]);
            data = permute(data,[1 3 2]);
            eventInterval = diff(latency(:,1))/obj.samplingRate;
        end
        %%
        function segObj = segmenting(obj,startEventMarkers,endEventMarkers, segmentName, channels)
            % Creates a new object containing only the data within the
            % specified segments. Between every two segments a boundary event
            % is inserted.
            %
            % Input arguments:
            %       startEventMarkers: string or cell array of strings with
            %                          the event marker signalizing the start
            %                          of the segment
            %       endEventMarkers:   string or cell array of strings with the
            %                          event markers signalizing the end of the
            %                          segment
            %       segmentName:       string to be added to the name of the new
            %                          object (optional), default "seg"
            %       channels:          indices of the channels to include in the
            %                          segmented object, default: all
            %        
            % Output argument:
            %       segObj:            segmented object
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       segObj = eegObj.segmenting({'701'},{'702'}, 'walking');
            %       % where '701' -> 'Go' and '702'-> Stop event markers

            if nargin < 3, error('Not enough input arguments.');end
            if nargin < 4, segmentName = 'seg';end
            if nargin < 5, channels = 1:obj.numberOfChannels;end
            if ischar(startEventMarkers), startEventMarkers = {startEventMarkers};end
            if ischar(endEventMarkers),   endEventMarkers   = {endEventMarkers};  end
            bsObj = basicSegment(obj,startEventMarkers,endEventMarkers,segmentName);
            segObj = bsObj.apply(obj,channels);
            parentCommand.uuid = obj.uuid; %#ok
            parentCommand.commandName = 'segmenting';%#ok
            parentCommand.varargin = {startEventMarkers,endEventMarkers, segmentName, channels};%#ok
            segObj.parentCommand = parentCommand;%#ok
        end
        %%
        function cobj = divideStreamObject(obj,channel,label,name)
            % Creates a new object from a subset of channels in the parent
            % object.
            % 
            % Input arguments:
            %       channels: channels to keep
            %       labels:     labels of the channels in the new object, if
            %                   passed empty uses the labels in the parent 
            %                   object
            %       name:       name of the new object, default: 'subSet_' +
            %                   obj.name
            %        
            % Output arguments:
            %       cobj:       handle to the new object
            %
            % Usage:
            %       eegObj     = mobilab.allStreams.item{ eegItem };
            %       channels   = [1:2 3 4:10 45 ... ]; % channels to select
            %       labels     = eegObj.label( channels );
            %       objectName = 'eeg2';
            %       eegObj2    = ...
            %       eegObj.divideStreamObject(channels,labels,objectName);

            if nargin < 3, label = obj.label(channel);end
            if isempty(label), label = obj.label(channel);end
            if length(label) ~= length(channel), error('Input arguments ''channel'' and ''label'' must have the same length.');end
            if nargin < 4, name = ['subSet_' obj.name];end
            commandHistory.commandName = 'divideStreamObject';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = channel;
            commandHistory.varargin{2} = label;
            commandHistory.varargin{3} = name;
            
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            metadata.uuid = generateUUID;
            path = fileparts(obj.binFile);
            
            try metadata.name = name;
                metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                metadata.numberOfChannels = length(commandHistory.varargin{1});
                metadata.label = commandHistory.varargin{2};
                channels = commandHistory.varargin{1};
                if isfield(metadata,'channelSpace'),
                    try   metadata.channelSpace = metadata.channelSpace(channels,:);
                    catch metadata.channelSpace = [];
                    end
                end
            catch ME
                ME.rethrow;
            end
            fid = fopen(metadata.binFile,'w');
            if fid<=0, error('Cannot create a new file. You probably ran out of hdd space.');end;
            for it=1:metadata.numberOfChannels, fwrite(fid,obj.mmfObj.Data.x(:,channels(it)),obj.precision);end
            fclose(fid);
            newHeader = metadata2headerFile(metadata);
            cobj = obj.container.addItem(newHeader);
        end
        %%
        function initStatusbar(obj,mn,mx,msg)
            if nargin < 4, error('Not enough input arguments.');end
            try obj.container.initStatusbar(mn,mx,msg);end %#ok
        end
        %%
        function statusbar(obj,val,msg)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3
                 obj.container.statusbar(val);
            else obj.container.statusbar(val,msg);
            end
        end
        %%
        function val     = isGuiActive(obj), val     = obj.container.isGuiActive;end
        function hFigure = refreshGui(obj),  hFigure = obj.container.refreshGui;end
    end
    methods(Hidden = true)
        %%
        function val = retrieveProperty(obj,propName) %#ok
            load(obj.header,'-mat',propName);
            if ~exist(propName,'var'), error('MoBILAB:unknownProperty',['Unknown variable ''' propName ''' in header.']);end
            eval(['val=' propName ';']);
        end
        function saveProperty(obj,propName,propValue) %#ok
            if ~isprop(obj,propName), error(['Unknown variable ''' propName ''' in header.']);end
            eval([propName '=propValue;']);
            disp(['Saving ' propName ' in: ' obj.header]);
            save(obj.header,'-mat','-append',propName);
        end
        %%
        function connect(obj)
            % Creates the connection between the memory mapping object and 
            % the binary file.
            
            try descriptor = dir(obj.binFile);
                dim = [length(obj.timeStamp) obj.numberOfChannels];
                if ~isempty(obj.binFile) && ~isempty(obj.precision) && dim(2) > 0 && ~isempty(descriptor) && descriptor.bytes
                    obj.mmfObj = memmapfile(obj.binFile,'Format',{obj.precision dim 'x'},'Writable',obj.writable);
                end
            catch obj.mmfObj = [];%#ok
            end
        end
        %%
        function disconnect(obj)
            % Destroy the memory mapping object closing the connection to
            % the binary file.
            
            obj.mmfObj = [];
        end  
        %%
        function disp(obj)
            % Returns basic information about the object and its fields.
            
            if ~isvalid(obj); disp('Invalid or deleted object.');return;end
            dim = obj.size;
            auxNch = length(obj.auxChannel.label);
            string = sprintf('\nClass:  %s\nProperties:\n  name:                 %s\n  uuid:                 %s\n  samplingRate:         %i Hz\n  timeStamp:            <1x%i double>\n  numberOfChannels:     %i\n  data:                 <%ix%i %s>\n  event.latencyInFrame: <1x%i double>\n  event.label:          <%ix1 cell>',...
                class(obj),obj.name,char(obj.uuid),obj.samplingRate,length(obj.timeStamp),obj.numberOfChannels,dim(1),dim(2),obj.precision,length(obj.event.latencyInFrame),length(obj.event.label));
            
            if iscellstr(obj.unit) && ~isempty(obj.unit{1})
                 unit = obj.unit{1}; %#ok
            else unit = 'none';      %#ok
            end
            string = sprintf('%s\n  label:                <%ix1 cell>',string, dim(2));
            string = sprintf('%s\n  unit:                 %s',string, unit); %#ok
            string = sprintf('%s\n  sessionUUID:          %s',string,char(obj.sessionUUID));
            string = sprintf('%s\n  auxChannel.label:     <%ix1 cell>',string, auxNch);
            string = sprintf('%s\n  auxChannel.data:      <%ix%i %s>',string, dim(1),auxNch,obj.precision);
            try %#ok
                string = sprintf('%s\n  history:              %s',string,obj.history(1,:));
                for it=2:size(obj.history,1)
                    string = sprintf('%s\n                        %s',string,obj.history(it,:));
                end
            end
            disp(string);
        end
        %%
        function properyArray = getPropertyGridField(obj)
            dim = size(obj);
            L = '{';
            for k=1:obj.numberOfChannels
                L = [L '''' obj.label{k} ''','];
            end
            L(end) = '}';
            if ~isempty(obj.event.uniqueLabel)
                U = '{';
                for k=1:length(obj.event.uniqueLabel)
                    U = [U '''' obj.event.uniqueLabel{k} ''','];
                end
                U(end) = '}';
            else
                U = '{}';
            end
            if ~isempty(obj.auxChannel.label)
                A = '{';
                for k=1:length(obj.auxChannel.label)
                    A = [A '''' obj.auxChannel.label{k} ''','];
                end
                A(end) = '}';
            else
                A = '{}';
            end
            properyName = {'class','name','uuid','sessionUUID','samplingRate','timeStamp','numberOfChannels','data','label','eventLabels','history','auxChannelLabel','auxChannelData'};
            properyVal = {class(obj),obj.name,obj.uuid,obj.sessionUUID,num2str(obj.samplingRate),['<1x' num2str(dim(1)) ' ' class(obj.timeStamp) '>'],num2str(obj.numberOfChannels),...
                ['<' num2str(dim(1)) 'x' num2str(dim(2)) ' ' obj.precision '>'],L,U,obj.history,A,...
                ['<' num2str(dim(1)) 'x' num2str(size(obj.auxChannel.data,2)) ' ' obj.precision '>']};
            properyArray = {properyName,properyVal};
        end
        %%
        function metadata = saveobj(obj)
            metadata = load(obj.header,'-mat');
            metadata.class = class(obj);
            if isfield(metadata,'history'), metadata = rmfield(metadata,'history');end
        end
        %%
        function history = serializeCommand(obj)
            dSname = 'mobilab.allStreams';
            if isfield(obj.parentCommand,'uuid')
                if strcmp(obj.parentCommand.commandName,'apply')
                    segIndex = obj.container.segmentList.findItem(obj.parentCommand.segmentName);
                    itemIndex = obj.container.findItem(obj.parentCommand.uuid);
                    %
                    tmp = obj.container.segmentList.item{segIndex}.history;
                    loc1 = find(tmp=='(');
                    loc2 = find(tmp==')');
                    history = [dSname '.segmentList.addSegment(' tmp(loc1+1:loc2-1) ');'];    
                    tmp = [dSname '.segmentList.item{' num2str(segIndex) '}.apply(' dSname '.item{' num2str(itemIndex)...
                        '}, [' num2str(obj.parentCommand.channels(:)') ']);'];
                    history = char(cat(1,{history},{tmp}));
                    return;
                else
                    itemIndex = obj.container.findItem(obj.parentCommand.uuid);
                    history = [dSname '.item{' num2str(itemIndex) '}.' obj.parentCommand.commandName '('];
                end
            else
                history = [dSname '=' obj.parentCommand.commandName '('];
            end
            if ~isfield(obj.parentCommand,'varargin')
                history(end) = ';';
                return;
            end
            for it=1:length(obj.parentCommand.varargin)
                if ischar(obj.parentCommand.varargin{it})
                    history = [history '''' obj.parentCommand.varargin{it} ''',']; %#ok
                elseif isvector(obj.parentCommand.varargin{it}) && length(obj.parentCommand.varargin{it}) > 1 && iscellstr(obj.parentCommand.varargin{it})
                    cellOfLabels = '{';
                    for jt=1:length(obj.parentCommand.varargin{it})
                        cellOfLabels = [cellOfLabels '''' obj.parentCommand.varargin{it}{jt} ''' '];%#ok
                    end
                    cellOfLabels(end) = '}';
                    history = [history ' ' cellOfLabels ','];%#ok
                elseif isvector(obj.parentCommand.varargin{it}) && length(obj.parentCommand.varargin{it}) > 1 && isnumeric(obj.parentCommand.varargin{it})
                    history = [history '[' num2str(obj.parentCommand.varargin{it}(:)') '],'];%#ok
                elseif isnumeric(obj.parentCommand.varargin{it})
                    history = [history  ' [' num2str(obj.parentCommand.varargin{it}(:)') '] ,'];%#ok
                elseif islogical(obj.parentCommand.varargin{it})
                    history = [history  num2str(obj.parentCommand.varargin{it}) ','];%#ok
                else
                    history = [history  'arg' num2str(it) ','];%#ok
                end
            end
            history = [history(1:end-1) ');'];
        end
        %%
        function jsonObj = getJSON(obj)
            warning('MoBILAB:deprecated','This function is been deprecated, next time use ''serialize''.');
            jsonObj = serialize(obj);
        end
        %%
        function addSamples(obj,sample,timeStamp,eventChannel)
            checkNumberInputArguments(2, 4)
            if ~isempty(sample)
                try
                    sample = eval([obj.precision '(sample);']);
                catch ME
                    ME.rethrow;
                end
                fid = fopen(obj.binFile,'a');
                if fid < 0, error('prog:input','Error adding samples.\nThe file could not be opened.'), end
                fwrite(fid,sample(:),obj.precision);
                fclose(fid);
            end
            if nargin == 4
                obj.event = obj.event.addEventFromChannel(eventChannel,length(obj.timeStamp));
                N = length(timeStamp);
                obj.timeStamp(end+1:end+N) = timeStamp;
            end
        end
        %%
        function correctTimeStampDefects(obj,t0,samplingRate)
            if nargin < 2, error('Not enough input arguments.');end
            try 
                obj.disconnect;
                obj.timeStamp = t0(:)';
                if nargin > 2
                    obj.samplingRate = obj.samplingRate;
                    saveProperty(obj,'samplingRate',samplingRate);
                end
                obj.connect;
            catch  ME
                ME.throw;
            end
        end
    end
    %%
    methods(Abstract)
        jmenu = contextMenu(obj)
        newHeader = createHeader(obj,commandHistory)
        h = plot(obj);
    end
end