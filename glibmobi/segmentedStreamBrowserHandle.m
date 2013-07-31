classdef segmentedStreamBrowserHandle < streamBrowserHandle
    properties
        segmentObj
        segmentedStreamObj
        lightColor
        gObjHandle2
        channelIndex2
    end
    methods
        %% constructor
        function obj = segmentedStreamBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.originalStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.originalStreamObj.timeStamp(end);
                defaults.gain = 0.25;
                defaults.normalizeFlag = false;
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.originalStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.originalStreamObj.timeStamp(end);end
            if ~isfield(defaults,'gain'), defaults.gain = 0.25;end
            if ~isfield(defaults,'normalizeFlag'), defaults.normalizeFlag = false;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = false;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            obj@streamBrowserHandle(dStreamObj.originalStreamObj,defaults);
            obj.segmentedStreamObj = dStreamObj;
            type = dStreamObj.event.label;
            latency = dStreamObj.originalStreamObj.getTimeIndex(dStreamObj.timeStamp(dStreamObj.event.latencyInFrame));
            obj.eventObj = obj.eventObj.addEvent(latency,type);
            obj.initOsdColor;
            hListbox = findobj(obj.figureHandle,'Tag','listbox1');
            set(hListbox,'Value',1);
            set(hListbox,'String',obj.eventObj.uniqueLabel);
            obj.lightColor = [0.9412 0.9412 0.9412];%obj.lightColor = [0.8314 0.8157 0.7843];
            obj.segmentObj = dStreamObj.segmentObj;
            obj.uuid = dStreamObj.uuid;
            obj.plotThisTimeStamp(obj.nowCursor);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@streamBrowserHandle(obj);
            defaults.browserType = 'segmentedDataStreamBrowser';
            defaults.lightColor = obj.lightColor;
        end
        %%
        function obj = changeSettings(obj)
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
            obj.eventObj = obj.streamHandle.event;
            type = obj.segmentedStreamObj.event.label;
            latency = obj.streamHandle.getTimeIndex(obj.segmentedStreamObj.timeStamp(obj.segmentedStreamObj.event.latencyInFrame));
            obj.eventObj = obj.eventObj.addEvent(latency,type);
            obj.initOsdColor;
            
             prefObj = [...
                PropertyGridField('gain',obj.gain,'DisplayName','Channel gain','Description','')...
                PropertyGridField('channels',1:obj.streamHandle.numberOfChannels,'DisplayName','Channels to plot','Description','This field accepts matlab code returning a subset of channels, for instance use: ''setdiff(1:10,[3 5 7])'' to plot channels from 1 to 10 excepting 3, 5, and 7.')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                PropertyGridField('normalizeFlag',obj.normalizeFlag,'DisplayName','Normalize channels','Description','Divides each channels by its standard deviation within the segment to plot.')...
                PropertyGridField('showChannelNumber',obj.showChannelNumber,'DisplayName','Show channel number or label','Description','')...
                PropertyGridField('onscreenDisplay',obj.onscreenDisplay,'Category','Events','DisplayName','Show events','Description','')...
                PropertyGridField('labels', obj.eventObj.uniqueLabel,'Category','Events','DisplayName','Show only a subset of events','Description','')...
                PropertyGridField('colormap',obj.colormap,'Type',PropertyType('char','row',{'lines','eegplot'}),'DisplayName','Colormap','Description','')...
                ];
            
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            
            if ~isempty(val.labels)
                rmLabels = setdiff(obj.eventObj.uniqueLabel,val.labels);
                if ~isempty(rmLabels)
                    for it=1:length(rmLabels), obj.eventObj = obj.eventObj.deleteAllEventsWithThisLabel(rmLabels{it});end
                end
            end
            

            obj.gain = val.gain;
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.windowWidth = val.windowWidth;
            obj.normalizeFlag = val.normalizeFlag;
            obj.showChannelNumber = val.showChannelNumber;
            obj.channelIndex = val.channels;
            obj.onscreenDisplay = val.onscreenDisplay;
            obj.changeColormap(val.colormap);
            
            figure(obj.figureHandle);
            obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
            obj.createGraphicObjects(obj.nowCursor);
            
            if isa(obj.master,'browserHandleList')
                obj.master.bound = max([obj.master.bound obj.windowWidth]);
                obj.master.nowCursor = obj.windowWidth/2;
                for it=1:length(obj.master.list)
                    if obj.master.list{it} ~= obj
                        obj.master.list{it}.nowCursor = obj.master.nowCursor;
                    end
                end
                obj.master.plotThisTimeStamp(obj.master.nowCursor);
            end
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
           plotThisTimeStamp@streamBrowserHandle(obj,nowCursor);
           if isempty(obj.segmentObj), return;end
           set(obj.gObjHandle,'Color',obj.lightColor);
           XData = get(obj.gObjHandle,'XData');
           time = XData{1};
           I = false(1,length(time));
           for it=1:length(obj.segmentObj.startLatency), I = I | (time >= obj.segmentObj.startLatency(it) & time <= obj.segmentObj.endLatency(it));end
           try delete(obj.gObjHandle2);end %#ok
           obj.gObjHandle2 = [];
           if any(I),
               
               YData = get(obj.gObjHandle,'YData');
               data = cell2mat(YData);
               hold(obj.axesHandle,'on');
               for it=1:length(obj.segmentObj.startLatency)
                   I = (time >= obj.segmentObj.startLatency(it) & time <= obj.segmentObj.endLatency(it));
                   if any(I), obj.gObjHandle2 = [obj.gObjHandle2; plot(obj.axesHandle,time(I),data(:,I))];end
               end               
               hold(obj.axesHandle,'off');
           end
        end
    end
end