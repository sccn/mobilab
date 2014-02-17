classdef spectrogramBrowserHandle < browserHandle
    properties
        gObjHandle
        windowWidth
        dbFlag
        osdColor  % onscreen display color
        textHandle
        eventObj
        zoomHandle
        axesSize
        channelIndex
        clim
    end
    methods
        %% constructor
        function obj = spectrogramBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.dbFlag = true;
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.channels = 1;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
                defaults.font = struct('size',12,'weight','normal');
                defaults.gui = @streamBrowserNG;
                defaults.onscreenDisplay = true;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'dbFlag'), defaults.dbFlag = true;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'channels'), defaults.channels = 1;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if defaults.windowWidth > defaults.endTime - defaults.startTime, defaults.windowWidth = defaults.endTime - defaults.startTime; end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'gui'), defaults.gui = @streamBrowserNG;end
            if ~isfield(defaults,'onscreenDisplay'), defaults.onscreenDisplay = true;end
            
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('timeIndex','PostSet',@streamBrowserHandle.updateTimeIndexDenpendencies);
            obj.addlistener('font','PostSet',@streamBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.cursorHandle = [];
            obj.state = false;
            obj.dbFlag = defaults.dbFlag;
            obj.textHandle = [];
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.onscreenDisplay = defaults.onscreenDisplay; % onscreen display information (e.g. events, messages, etc)
            
            obj.channelIndex = defaults.channels; 
            obj.eventObj = obj.streamHandle.event;
            obj.initOsdColor;
            
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = defaults.gui(obj);
        end
        %%
        function initOsdColor(obj)
            tmpColor = lines(length(obj.eventObj.uniqueLabel));
            obj.osdColor = zeros(length(obj.eventObj.latencyInFrame),3);
            for it=1:length(obj.eventObj.uniqueLabel)
                loc = ismember(obj.eventObj.label,obj.eventObj.uniqueLabel(it));
                obj.osdColor(loc,1) = tmpColor(it,1);
                obj.osdColor(loc,2) = tmpColor(it,2);
                obj.osdColor(loc,3) = tmpColor(it,3);
            end
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.windowWidth = obj.windowWidth;
            defaults.dbFlag = obj.dbFlag;
            defaults.osdColor = obj.osdColor;
            defaults.browserType = 'streamBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.figureHandle,'RendererMode','manual')
            set(obj.figureHandle,'Renderer','painters')
            set(obj.axesHandle,'drawmode','fast');
            view(obj.axesHandle,[0 90]);
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2))); 
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor+obj.windowWidth/2));
            % data = obj.streamHandle.power(obj.timeIndex(t1:t2),:,obj.channelIndex);            
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),:,obj.channelIndex).^2 +...
                obj.streamHandle.mmfObj.Data.y(obj.timeIndex(t1:t2),:,obj.channelIndex).^2;
            
            data(data==0) = eps;
            Data = obj.streamHandle.mmfObj.Data.x(1:100:end,:,obj.channelIndex).^2 + obj.streamHandle.mmfObj.Data.y(1:100:end,:,obj.channelIndex).^2;
            Data(Data==0) = eps;
            obj.clim(1) = prctile(10*log10(Data(:)),99)/10;
            obj.clim(2) = obj.clim(1)*10;
            if obj.clim(1) <=0
                obj.clim = [prctile(10*log10(Data(:)),10) prctile(10*log10(Data(:)),99)];
            end
            % fs = obj.streamHandle.samplingRate;
            % k = 2;
            % bLatency = obj.streamHandle.event.getLatencyForEventLabel('boundary');
            % I = [];for it=1:length(bLatency), I = [I bLatency(it)-k*fs:bLatency(it)+k*fs];end %#ok
            % nBoundary = setdiff(1:size(obj.streamHandle,1),I);
            % mn = min(Data(nBoundary,:,obj.channelIndex),[],2);
            % mx = max(Data(nBoundary,:,obj.channelIndex),[],2);
            
            % obj.dbFlag = false;
            if obj.dbFlag
                data = 10*log10(data);
                titleStr = 'Power Spectral Density (dB)';
            else
                obj.clim = power(10,obj.clim/10);
                titleStr = 'Power Spectral Density';
            end
            titleStr = [titleStr '  Channel: ' obj.streamHandle.label{obj.channelIndex}];
            data = data - min(data(:));
            data = data/(max(data(:)));
            data = data*diff(obj.clim);
            data = data + obj.clim(1);
            
            cla(obj.axesHandle);
            obj.gObjHandle = imagesc(obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),log10(obj.streamHandle.frequency),data','Parent',obj.axesHandle);
            
            tick = get(obj.axesHandle,'Ytick');
            fval = 10.^tick;
            Nf = length(tick);
            yLabel = cell(Nf,1);
            fval(fval >= 10) = round(fval(fval >= 10));
            for it=1:Nf, yLabel{it} = num2str(fval(it),3);end
            set(obj.axesHandle,'YDir','normal','Ytick',tick,'YTickLabel',yLabel,'CLim',obj.clim);
          
            ylabel(obj.axesHandle,'Frequency (Hz)');
            title(obj.axesHandle,titleStr)
            % set(obj.axesHandle,'YDir','normal','YLim',obj.streamHandle.frequency([1 end]),'YScale','log','CLim',obj.clim);
            % set(obj.axesHandle,'YDir','normal','YLim',obj.streamHandle.frequency([1 end]),'YScale','log');
            % set(obj.gObjHandle,'CDataMapping','direct');
            obj.plotThisTimeStamp(nowCursor);
            
            val = linspace(1.1*obj.clim(1),0.9*obj.clim(2),6);
            label = cell(6,1);
            for it=1:6, label{it} = num2str(val(it),4);end
            colorbar('YTickLabel',label);
            % colorbar;
            
            try delete(obj.cursorHandle.gh);end %#ok
            obj.cursorHandle.ghIndex = 1;%floor(obj.numberOfChannelsToPlot/2+1);
            obj.cursorHandle.gh = graphics.cursorbar(obj.axesHandle,'Parent',obj.axesHandle);
            obj.cursorHandle.gh.CursorLineColor = 'r';%[.9,.3,.6]; % default=[0,0,0]='k'
            obj.cursorHandle.gh.CursorLineStyle = '-.';       % default='-'
            obj.cursorHandle.gh.CursorLineWidth = 2.5;        % default=1
            obj.cursorHandle.gh.Orientation = 'vertical';     % =default
            obj.cursorHandle.gh.TargetMarkerSize = 12;        % default=8
            obj.cursorHandle.gh.TargetMarkerStyle = 'none';      % default='s' (square)
            set(obj.cursorHandle.gh.BottomHandle,'MarkerSize',8)
            set(obj.cursorHandle.gh.TopHandle,'MarkerSize',8)
            obj.cursorHandle.gh.visible = 'off';
            set(obj.cursorHandle.gh,'UpdateFcn',@updateCursor);
            obj.cursorHandle.gh.ShowText = 'on';
            obj.cursorHandle.gh.Tag = 'graphics.cursorbar';
            set(get(obj.cursorHandle.gh,'DisplayHandle'),'Visible','off')
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
            delta = obj.windowWidth/2;
            if  nowCursor + delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    nowCursor - delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = nowCursor;
            elseif nowCursor + delta >= obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end)) - delta;
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            else
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + delta;
            end
            nowCursor = newNowCursor;
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2))); 
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor+obj.windowWidth/2));
            if t1==t2, return;end
            
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),:,obj.channelIndex).^2 +...
                obj.streamHandle.mmfObj.Data.y(obj.timeIndex(t1:t2),:,obj.channelIndex).^2;
            if obj.dbFlag, data = 10*log10(data);end
            
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                try delete( findall(obj.axesHandle,'Tag','graphics.cursorbar'));end %#ok
            end
            data(isinf(data)) = nan;
            set(obj.gObjHandle,'CData',data','XData',obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));           
            xlim(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));
                        
            if obj.onscreenDisplay
                if length(obj.eventObj.latencyInFrame) ~= size(obj.osdColor,1), obj.initOsdColor;end
                [~,loc1,loc2] = intersect(obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));
                Nloc1 = length(loc1);
                if ~isempty(loc1)
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    ind = obj.timeIndex(t1:t2);
                    lim = get(obj.axesHandle,'Ylim');
                    linesHandler = line(ones(2,1)*obj.streamHandle.timeStamp(ind(loc1)),(ones(length(loc1),1)*lim)','Parent',obj.axesHandle);
                    %kkIndex = ind(loc1) == 1;
                    
                    textPos = [obj.streamHandle.timeStamp(ind(loc1))-0.5*(obj.streamHandle.timeStamp(ind(loc1))-...
                            obj.streamHandle.timeStamp(ind(loc1)-0));ones(1,Nloc1)*lim(2)*1.01];

                    try delete(obj.textHandle);end %#ok
                    set(linesHandler,{'color'},num2cell(obj.osdColor(loc2,:)',[1 3])');
                    obj.textHandle = zeros(length(loc1),1);
                    for it=1:Nloc1
                        obj.textHandle(it) = text('Position',textPos(:,it),'String',obj.eventObj.label(loc2(it)),'Color',obj.osdColor(loc2(it),:),...
                            'Parent',obj.axesHandle,'FontSize',12,'FontWeight','bold','Rotation',45);
                    end
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    hold(obj.axesHandle,'off');
                end
            end
            
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                obj.cursorHandle.gh = graphics.cursorbar(obj.axesHandle,'Parent',obj.axesHandle);
                obj.cursorHandle.gh.CursorLineColor = 'r';%[.9,.3,.6]; % default=[0,0,0]='k'
                obj.cursorHandle.gh.CursorLineStyle = '-.';       % default='-'
                obj.cursorHandle.gh.CursorLineWidth = 2.5;        % default=1
                obj.cursorHandle.gh.Orientation = 'vertical';     % =default
                obj.cursorHandle.gh.TargetMarkerSize = 12;        % default=8
                obj.cursorHandle.gh.TargetMarkerStyle = 'none';      % default='s' (square)
                set(obj.cursorHandle.gh.BottomHandle,'MarkerSize',8)
                set(obj.cursorHandle.gh.TopHandle,'MarkerSize',8)
                obj.cursorHandle.gh.visible = 'on';
                set(obj.cursorHandle.gh,'UpdateFcn',@updateCursor);
                obj.cursorHandle.gh.ShowText = 'on';
                obj.cursorHandle.gh.Tag = 'graphics.cursorbar';
                set(get(obj.cursorHandle.gh,'DisplayHandle'),'Visible','off')
            end
            
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            
        end
        %%
        function plotStep(obj,step)
            delta = obj.windowWidth/2;
            if obj.nowCursor+step+delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    obj.nowCursor+step-delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step+delta > obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end))-delta;
            else
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1))+delta;
            end
            obj.plotThisTimeStamp(newNowCursor);
        end
        %%
        function delete(obj)
            try close(get(obj.axesHandle,'userData'));end %#ok
            try delete(obj.figureHandle);end %#ok
            if ~isa(obj.master,'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
        end
        %%
        function obj = changeSettings(obj)
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
             prefObj = [...
                PropertyGridField('channel',1:obj.streamHandle.numberOfChannels,'DisplayName','Channel to plot')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                PropertyGridField('onscreenDisplay',obj.onscreenDisplay,'Category','Events','DisplayName','Show events','Description','')...
                PropertyGridField('dbFlag',obj.dbFlag,'DisplayName','Plot in dB')...
                PropertyGridField('labels', obj.streamHandle.event.uniqueLabel,'Category','Events','DisplayName','Show only a subset of events','Description','')...
                ];
            %PropertyGridField('dbFlag',obj.dbFlag,'DisplayName','Plot in dB')...
            
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            
            
            obj.eventObj = event;
            if ~isempty(val.labels)
                for it=1:length(val.labels)
                    latency = obj.streamHandle.event.getLatencyForEventLabel(val.labels{it});
                    obj.eventObj = obj.eventObj.addEvent(latency,val.labels{it});
                end
            end

            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.windowWidth = val.windowWidth;
            obj.dbFlag = val.dbFlag;
            obj.channelIndex = val.channel(1);
            obj.onscreenDisplay = val.onscreenDisplay;
            
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
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);
            if ~any(I), return;end
            obj.channelIndex = channelIndex(1);
        end
    end
    %%
    methods(Static)
        function updateTimeIndexDenpendencies(~,evnt)
            if evnt.AffectedObject.timeIndex(1) ~= -1
                evnt.AffectedObject.nowCursor = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)) + 2.5;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
        %%
        function updateFont(~,evnt)
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
    end
end