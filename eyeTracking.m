%% eyeTracking class
% Creates and uses a mocap object using predefined rigidbodies in the mocap data.
%
% Author: Marius Klug, TU Berlin, 13-Sep-2016 adapted from Alejandro Ojeda

%%
classdef eyeTracking < dataStream
    properties(GetAccess = public, SetAccess = public)
%         animationParameters; % Structure used for animating stick figures.
%         % It should have the following fields:
%         % limits: [min _x max_x; min_y max_y; min_z max_z]
%         %         specifying the dimensions of the mocap space
%         % conn:   [marker_i marker_j] matrix specifying connections
%         %         between markers.
%         
%         lsMarker2JointMapping
    end
    properties(GetAccess = public, SetAccess = protected, AbortSet = true)
%         bodyModelFile
    end
    properties(Dependent)
%         dataInXYZ           % Dependent property that reshapes the second dimension
        % of the field data to allows accessing directly xyz
        % coordinates of motion capture markers.
        
%         magnitude           % Dependent property that computes the magnitude (distance
        % from the origin) of xyz motion capture markers.
    end
    methods
        %%
        function obj = eyeTracking(header)
            % Creates a mocap object.
            %
            % Input arguments:
            %       header: header file (string)
            %
            % Output arguments:
            %       obj: mocap object (handle)
            
            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        
        %%
        function jsonObj = serialize(obj)
            metadata = saveobj(obj);
            metadata.class = class(obj);
            metadata.size = size(obj);
            metadata.event = obj.event.uniqueLabel;
            metadata.artifactMask = sum(metadata.artifactMask(:) ~= 0);
            metadata.writable = double(metadata.writable);
            metadata.history = obj.history;
            if isempty(metadata.animationParameters.conn)
                metadata.hasStickFigure = 'no';
            else
                metadata.hasStickFigure = 'yes';
            end
            metadata = rmfield(metadata,'animationParameters');
            metadata = rmfield(metadata,{'parentCommand' 'timeStamp','hardwareMetaData'});
            jsonObj = savejson('',metadata,'ForceRootName', false);
        end
       
        %%
        function cobj = lowpass(obj, varargin)
            % Filters the motion capture data with a zero-lag lowpass FIR
            % filter calling the method filter defined in dataStream.
            %
            % Input arguments:
            %       cutOff:   lowpass cutoff frequency (in Hz)
            %       channels: channel to filter, default: all
            %
            % Output argument:
            %       cobj:      handle to the new object
            %
            % Usage:
            %       mocapObj     = mobilab.allStreams.item{ mocapItem };
            %       cutOff       = 6;  % lowpass at 6 Hz
            %       mocapObjFilt = mocapObj.lowpass( cutOff );
            %
            %       figure;plot(mocapObj.timeStamp, [mocapObj.data(:,1) mocapObjFilt.data(:,1)])
            %       xlabel('Time (sec)');legend({mocapObj.name mocapObjFilt.name});
            
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter the cutoff frequency:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(3)};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = str2double(varargin{1});
                if isnan(varargin{1}), return;end
                dispCommand = true;
            end
            
            % Cutoff Frequency
            if nargin < 2, fc = 6; else fc = varargin{1};end
            if nargin < 3, channels = 1:obj.numberOfChannels; else channels = varargin{2};end
            if nargin < 4
                N = 128;
                disp('Third argument must be the length of the filter (integer type). Using the default: 128.');
            elseif isnumeric(varargin{3})
                N = varargin{3};
            else N = 128;
            end
            try cobj = obj.filter('lowpass',fc,channels,N);
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        
        %%
        function cobj = oneEuroFilter(obj, varargin)
            % Filters the motion capture data with a one euro filter. 
            % See G?ry Casiez, Nicolas Roussel, Daniel Vogel.  1? Filter:  A Simple Speed-based Low-pass Filter for
            % Noisy Input in Interactive Systems.  CHI?12, the 30th Conference on Human Factors in Computing
            % Systems, May 2012, Austin, United States. ACM, pp.2527-2530, 2012, <10.1145/2207676.2208639>.
            % <hal-00670496>
            %
            % Input arguments:
            %       mincutoff:   minimum lowpass cutoff frequency (in Hz)
            %       beta: beta value for adaptation of the filter -> higher value lead to less lag and more jitter for
            %       high frequency data
            %
            % Output argument:
            %       cobj:      handle to the new object
            %
            % Usage:
            %       mincutoff    = 1;
            %       beta         = 2;  
            %       mocapObjFilt = mobilab.allStreams.item{7}.oneEuroFilter( mincutoff , beta );
            
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter the minimal cutoff frequency:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(1.0)};
                inputFromDialog = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(inputFromDialog), return;end
                varargin{1} = str2double(inputFromDialog{1});
                if isnan(varargin{1}), return;end
                
                prompt = {'Enter the beta value:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(5.0)};
                inputFromDialog = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(inputFromDialog), return;end
                varargin{2} = str2double(inputFromDialog{1});
                if isnan(varargin{2}), return;end
                
                dispCommand = true;
            end
            
            % Cutoff Frequency and beta values
            if nargin < 1, mincutoff = 1.0; else mincutoff = varargin{1};end
            if nargin < 2, beta = 5.0; else beta = varargin{2};end

            try 

                
                noisySignal = obj.mmfObj.Data.x;
                commandHistory.commandName = 'oneEuroFilter';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = mincutoff;
                commandHistory.varargin{2} = beta;
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                %Declare oneEuro object
                theOneEuroFilter = oneEuro;
                %Alter filter parameters to tune
                theOneEuroFilter.mincutoff = mincutoff;
                theOneEuroFilter.beta = beta;
                
                filteredSignal = zeros(size(noisySignal));
                
                % filter all channels
                for channelToFilter = 1:size(noisySignal,2)
                    
                    % the filter goes through all data points 
                    for dataPoint = 1:size(noisySignal,1)
                        filteredSignal(dataPoint,channelToFilter) = theOneEuroFilter.filter(noisySignal(dataPoint,channelToFilter),dataPoint);
                    end
                    
                end
                
                % add the filtered data
                cobj.mmfObj.Data.x = filteredSignal;
                
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function cobj = timeDerivative(obj,varargin)
            % Computes the time derivatives of motion capture data. It smooths
            % the signals after each order of derivation to minimize cumulative
            % precision errors. As smoother it uses a zero-lag FIR lowpass filter.
            % Each new derivative is stored in a new object.
            %
            % Input arguments:
            %       order:     maximum order of derivation, default: 3 (1 = velocity,
            %                  2 = acceleration, 3 = jerk)
            %       cutOff:    lowpass filter cutoff, default: 18 Hz.
            %
            % Output argument:
            %       cobj:      handle to the object containing the latest order of
            %                  derivation
            %
            % Uses:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       order    = 3;
            %       mocapObj.timeDerivative( order );
            %
            %       figure;
            %       subplot(411);plot(mocapObj.timeStamp,mocapObj.data(:,1));xlabel('Time (sec)');title(mocapObj.name)
            %       subplot(412);plot(mocapObj.timeStamp, mocapObj.children{1}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{1}.name)
            %       subplot(413);plot(mocapObj.timeStamp, mocapObj.children{2}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{2}.name)
            %       subplot(414);plot(mocapObj.timeStamp, mocapObj.children{3}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{3}.name)
            
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && isnumeric(varargin{1}) && length(varargin{1}) == 1 && varargin{1} == -1
                prompt = {'Enter the order of the time derivative: ( 1->vel, 2->acc, 3->jerk,... or 1:3->for all of them)'};
                dlg_title = 'Time derivative input parameter';
                num_lines = 1;
                def = {num2str(obj.container.container.preferences.mocap.derivationOrder)};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = eval(varargin{1});
                dispCommand = true;
            end
            if nargin < 2, order = 3;else order = varargin{1};end
            if nargin < 3, fc = 6;  else fc = varargin{2};end
            if nargin < 4, channels = 1:obj.numberOfChannels;else channels = varargin{3};end
            if nargin < 5, filterOrder = 128;else filterOrder = varargin{4};end
            if ~isnumeric(order), error('prog:input','First argument must be the order of the derivative (1=veloc, 2=acc, 3=jerk).');end
            if ~isnumeric(fc),    error('prog:input','Second argument must be the cut off frequency.');end
            if ~isnumeric(channels), error('Invalid channel.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid channel.');end
            
            % checking for Quaternionvalues
            if sum(~cellfun('isempty',strfind(obj.label,'_A')))>0, error('Please transform quaternion values to euler angles before calculating time derivatives!');end
            
            Nch = length(channels);
            dt = 1/obj.samplingRate;
%             dt = 1e3*dt; % from seconds to mili seconds
            order = unique(1:max(order));
            N = max(order);
            try
                % smooth the data by 0 phase shifting moving average
                a = 1;
                b = obj.firDesign(filterOrder,'lowpass',fc);
                commandHistory.commandName = 'timeDerivative';
                commandHistory.uuid  = obj.uuid;
                commandHistory.varargin{2} = fc;
                commandHistory.varargin{3} = channels;
                obj.initStatusbar(1,N*Nch,'Computing time derivatives...');
                tmpObj = obj;
                tmpData = obj.mmfObj.Data.x;
                for derivative=1:N
                    for channel=1:size(tmpData,2)
                        
                        % deriving
                        tmpData(1:end-1,channel) = diff(tmpData(:,channel),1)/dt;
                        tmpData(end,channel) = tmpData(end-1,channel);
                        
                        % check if channel is Euler angles and if so,
                        % correct for turns over pi or -pi respectively
                        if strfind(obj.label{channel},'Euler')
                            
%                             cobj.mmfObj.Data.x(cobj.mmfObj.Data.x > 2*pi, jt) = cobj.mmfObj.Data.x(cobj.mmfObj.Data.x > 2*pi, jt) - 2*pi;
%                             cobj.mmfObj.Data.x(cobj.mmfObj.Data.x < -2*pi, jt) = cobj.mmfObj.Data.x(cobj.mmfObj.Data.x < -2*pi, jt) + 2*pi;

                            dataChannel = tmpData(:,channel);
                            
                            % turning rates of more than half a circle per frame are not possible
                            dataChannel(dataChannel > 180/dt) = dataChannel(dataChannel > 180/dt) - 2*180/dt;
                            dataChannel(dataChannel < -180/dt) = dataChannel(dataChannel < -180/dt) + 2*180/dt;
                            
                            tmpData(:,channel) = dataChannel;
                            
                        end
                        
                        % smoothing
%                       %  cobj.mmfObj.Data.x(:,jt) = filtfilt_fast(b,a,cobj.mmfObj.Data.x(:,jt));
                        obj.statusbar((Nch*(derivative-1)+channel));
                    end
                    
                    % creating the new object and filling the derived data
                    
                    commandHistory.varargin{1} = derivative;
                    cobj = tmpObj.copyobj(commandHistory);
                    
                    cobj.mmfObj.Data.x = tmpData;
                    
                    
                    % soft masking
                    
                    if derivative==1, artifactIndices = cobj.artifactMask(:) ~= 0;end
                    if any(artifactIndices), cobj.mmfObj.Data.x(artifactIndices) = cobj.mmfObj.Data.x(artifactIndices).*(1-cobj.artifactMask(artifactIndices));end
                    tmpObj = cobj;
                    
                end
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
            catch ME
                obj.statusbar(N*Nch);
                try obj.container.deleteItem(cobj.uuid);end %#ok
                ME.rethrow;
            end
        end
        
        %%
        function addEventsFromDerivatives(obj,varargin)

            dispCommand = false;
            if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
              
                dispCommand = true;
            end
            
            if dispCommand
                disp('Running:');
                disp('addEventsFromDerivatives');
            end
            
            for children = 1:size(obj.children,2)
               
                if ~isempty(strfind(obj.children{children}.name,'vel')) || ~isempty(strfind(obj.children{children}.name,'acc')) || ~isempty(strfind(obj.children{children}.name,'jerk')) || ~isempty(strfind(obj.children{children}.name,'Dt'))
                    
                    derivativeEvent = obj.children{children}.event;
                    obj.event = obj.event.addEvent(derivativeEvent.latencyInFrame,derivativeEvent.label);
               
                end
            end
            
            if dispCommand, disp('Done.');end
        end
        
        %%
        function deleteMarkers(obj,varargin)

            dispCommand = false;
            
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Which markers to keep?'};
                dlg_title = 'Input parameters';
                num_lines = 1;
                def = {''};
                markersToKeep = inputdlg2(prompt,dlg_title,num_lines,def)
                if isempty(varargin), return;end
                markersToKeep = strsplit(markersToKeep{1}, ' ')
                dispCommand = true;
            end
            
            command = 'Delete Markers, keep only:';
            for marker = 1:size(markersToKeep,1)
                
                command = strcat(command, {' '}, markersToKeep(marker));
                
            end
            
            if dispCommand
                disp('Running:');
                disp(command);
            end
            
            % determine for each unique marker if it is of the markers to keep or not
            for uniqueLabelInd = 1:size(obj.event.uniqueLabel,1)
               for markerInd = 1:size(markersToKeep,1)
                   
                    comparison(uniqueLabelInd,markerInd) = strcmp(obj.event.uniqueLabel{uniqueLabelInd},markersToKeep{markerInd});
                       
                end
            end
            comparison = sum(comparison,2);
            
            tempEvents = obj.event;
            % delete markers
            for uniqueLabelInd = 1:size(obj.event.uniqueLabel,1)
                
                if ~comparison(uniqueLabelInd)
                    
                    tempEvents = tempEvents.deleteAllEventsWithThisLabel(obj.event.uniqueLabel{uniqueLabelInd});
                    
                end
                
            end
            
            obj.event = tempEvents;
            
        end
        
        
        %%
        function createEventsFromMagnitude(obj,varargin)
            % criteria: 'maxima', 'minima', 'zero crossing', '% maxima', '% minima', or 'all' (default: criteria = 'all')
            % channel: index of the channel where to search for the events
            
            dispCommand = false;
            if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
                prefObj = [...
                    PropertyGridField('channel',4,'DisplayName','Channel','Category','Main','Description','Channel number. Enter desired channel to generate marker from and hit return.')...
                    PropertyGridField('criteria','movements','Type',PropertyType('char', 'row', {'maxima', 'minima','zero crossing', 'sliding window deviation', 'movements'}),'DisplayName','Criteria','Category','Main','Description','Criterion for making the event, could be: maxima, minima, zero crossing, sliding window deviation, movements.')...
                    PropertyGridField('correctSign',0,'DisplayName','Correct sign criteria','Category','Main','Description','Enter if max/min criteria should only be fulfilled if sign is pos/neg and hit return.')...
                    PropertyGridField('eventType','movement:start movement:end','DisplayName','Marker name','Category','Main','Description','Enter the name of the new event marker and hit return.')...
                    PropertyGridField('inhibitionWindow',2,'DisplayName','Inhibition/sliding window length','Category','Main','Description','Enter the length of the inhibition window and hit return. Is multiplied by sampling rate!')...
                    PropertyGridField('movementThreshold',65,'DisplayName','Threshold for detecting a general movement','Category','Main','Description','Percentage of values that should be lower than the threshold. Enter the threshold and hit return.')...
                    PropertyGridField('movementOnsetThresholdFine',5,'DisplayName','Threshold for detecting a movement onset in the velocity in percentage once a movement has been detected','Category','Main','Description','Enter the threshold and hit return.')...
                    PropertyGridField('minimumDuration',286,'DisplayName','Minimum duration for a movement','Category','Main','Description','If ''movement'' category is chosen, only movements longer than this duration (in ms) are considered. Enter and hit return.')...
                    ];
                
                hFigure = figure('MenuBar','none','Name','Create event marker','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
                    'Resize','off','userData',0);
                position = get(hFigure,'position');
                set(hFigure,'position',[position(1:2) 303 431]);
                hPanel = uipanel(hFigure,'Title','','BackgroundColor','white','Units','pixels','Position',[0 55 303 380],'BorderType','none');
                g = PropertyGrid(hPanel,'Properties', prefObj,'Position', [0 0 1 1]);%,'Description','Projects low-dimensional burst artifacts out of the data.');
                uicontrol(hFigure,'Position',[72 15 70 21],'String','Cancel','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@cancelCallback);
                uicontrol(hFigure,'Position',[164 15 70 21],'String','Ok','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@okCallback);
                uiwait(hFigure);
                if ~ishandle(hFigure), return;end
                if ~get(hFigure,'userData'), close(hFigure);return;end
                close(hFigure);
                drawnow
                val = g.GetPropertyValues();
                varargin{1} = val.channel;
                varargin{2} = val.criteria;
                varargin{3} = val.eventType;
                varargin{4} = val.inhibitionWindow;
                varargin{5} = val.correctSign;
                varargin{6} = val.movementThreshold;
                varargin{7} = val.movementOnsetThresholdFine;
                varargin{8} = val.minimumDuration;
                dispCommand = true;
            end
            
            Narg = length(varargin);
            if Narg < 1, channel   = 1;       else channel   = varargin{1}(1);end
            if Narg < 2, criteria  = 'maxima';else criteria  = varargin{2};   end
            if Narg < 3, eventType = [];else eventType = strsplit(varargin{3}, ' ');   end
            if Narg < 4
                inhibitedWindowLength = obj.samplingRate;
            else
                inhibitedWindowLength = ceil(obj.samplingRate*varargin{4});
            end
            if Narg < 5, correctSign = 0;   else correctSign = varargin{5}; end
            if Narg < 6, movementThreshold = 1.2;   else movementThreshold = varargin{6}; end
            if Narg < 7, movementOnsetThresholdFine = 0.05;   else movementOnsetThresholdFine = varargin{7} / 100; end
            if Narg < 8, minimumDuration = 0; else minimumDuration = varargin{8}; end
            if Narg < 9
                segmentObj = basicSegment([obj.timeStamp(1),obj.timeStamp(end)]);
            else
                segmentObj = varargin{8};
            end
            
            if dispCommand
                disp('Running:');
                disp('  latency = createEventsFromMagnitude(obj,mocap_marker, criteria, event_marker)');
            end
            
            numberOfSegments = length(segmentObj.startLatency);
            index = obj.getTimeIndex([segmentObj.startLatency segmentObj.endLatency]);
            index = reshape(index,numberOfSegments,2);
            
            if strcmp(eventType(1),'') || strcmp(eventType{1},'default')
                switch criteria
                    case 'maxima',                      eventType = {'max'};
                    case 'zero crossing',               eventType = {'zc'};
                    case 'minima',                      eventType = {'min'};
                    case 'sliding window deviation',    eventType = {'slideDev'};
                    case 'movements',                   eventType = {'onset', 'offset'};
                    otherwise,                          eventType = {'noname'};
                end
            end
            %signal = obj.magnitude(:,channel);
            signal = obj.mmfObj.data.x(:,channel);
            
            for it=1:numberOfSegments % default is 1 segment: the whole data stream
                [I J] = searchInSegment(signal(index(it,1):index(it,2)),criteria,inhibitedWindowLength,movementThreshold, movementOnsetThresholdFine, round(obj.samplingRate*minimumDuration/1000));
                if I == 0
                    break
                end
                if correctSign && (strcmp(criteria, 'maxima') || strcmp(criteria, 'minima'))
                    if strcmp(criteria, 'minima')
                        signal = -signal;
                    end
                    I(sign(signal(I))==-1)=[];
                end
                
                time = obj.timeStamp(index(it,1):index(it,2));
                latencyI = obj.getTimeIndex(time(I));
                 obj.event = obj.event.addEvent(latencyI,eventType{1});
                
                if J ~= 0
                    time = obj.timeStamp(index(it,1):index(it,2));
                    latencyJ = obj.getTimeIndex(time(J));
                    obj.event = obj.event.addEvent(latencyJ,eventType{2});
                end
                
            end
            if dispCommand, disp('Done.');end
        end
        
    end
    
    methods(Hidden=true)
       
        %%
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid        = obj.uuid;
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
               
                case 'throwOutChannels'
                    prename = 'throwOut_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    channels = commandHistory.varargin{1};
                    metadata.numberOfChannels = length(channels);
                    metadata.label = obj.label(channels);
                    metadata.artifactMask = obj.artifactMask(:,channels);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(channels)]);
                    
                case 'addChannels'
                    prename = 'addChan_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    channels = commandHistory.varargin{1};
                    metadata.numberOfChannels = length(channels);
                    metadata.label = obj.label(channels);
                    metadata.artifactMask = obj.artifactMask(:,channels);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(channels)]);
                
                case 'oneEuroFilter'
                    prename = 'oneEuroFilter_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = obj.numberOfChannels;
                    metadata.label = obj.label;
                    metadata.artifactMask = obj.artifactMask;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'timeDerivative'
                    order = commandHistory.varargin{1};
                    if ~isnumeric(order), error('Input parameter ''order'' must be a number indicating the order of the time derivative to calculate.');end
                    switch order
                        case 1, prename = 'vel_';
                        case 2, prename = 'acc_';
                        case 3, prename = 'jerk_';
                        otherwise
                            prename = ['Dt' num2str(order) '_'];
                    end
                    index = obj.container.findItem(commandHistory.uuid);
                    parentName = obj.container.item{ index }.name;
                    metadata.name = [prename parentName];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    if order == 1
                        channels = commandHistory.varargin{3};
                        fid = fopen(metadata.binFile,'w');
                        fwrite(fid,obj.mmfObj.Data.x(:,channels),obj.precision);
                        fclose(fid);
                        metadata.artifactMask = obj.artifactMask(:,channels);
                        metadata.numberOfChannels = length(channels);
                        metadata.label = obj.label(channels);
                    else
                        copyfile(obj.binFile,metadata.binFile,'f');
                    end
                    
                case 'save_as'
                    prename = '';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    uind = unique(obj.animationParameters.conn(:));
                    [~,loc_i] = ismember(obj.animationParameters.conn(:,1),uind) ;
                    [~,loc_j] = ismember(obj.animationParameters.conn(:,2),uind) ;
                    obj.reshape([length(obj.timeStamp) 3 obj.numberOfChannels/3]);
                    data = obj.mmfObj.Data.x(:,:,uind);
                    obj.reshape([length(obj.timeStamp) obj.numberOfChannels]);
                    metadata.animationParameters = obj.animationParameters;
                    metadata.animationParameters.conn = [loc_i loc_j];
                    tmp = uind*3;
                    channels = [];
                    for it=1:length(tmp)
                        channels(end+1) = tmp(it)-2;%#ok
                        channels(end+1) = tmp(it)-1;%#ok
                        channels(end+1) = tmp(it);%#ok
                    end
                    
                    fid = fopen(metadata.binFile,'w');
                    if fid<=0, error('Invalid file identifier. Cannot create the copy object.');end;
                    fwrite(data(:),obj.precision);
                    fclose(fid);
                    metadata.artifactMask = obj.artifactMask(:,channels);
                    N = length(channels);
                    metadata.label = cell(N,1);
                    for it=1:N, metadata.label{it} = obj.label{channels(it)};end
                    
                    
                otherwise
                    error('Cannot make a secure copy of this object. Please provide a valid ''commandHistory'' instruction.');
            end
            newHeader = metadata2headerFile(metadata);
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;

            menuItem = javax.swing.JMenuItem('Throw out channels');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'throwOutChannels',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Add channels');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'addChannels',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Lowpass filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'lowpass',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('One Euro filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'oneEuroFilter',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Compute time derivatives');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'timeDerivative',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Create event marker');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'createEventsFromMagnitude',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Add events from time derivative streams');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'addEventsFromDerivatives',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Delete Event Markers');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'deleteMarkers',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Inspect');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'inspect',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Annotation');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('Generate batch script');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@generateBatch_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="maroon">Delete object</HTML>');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});
        end
    end
end
