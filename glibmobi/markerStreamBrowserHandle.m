classdef markerStreamBrowserHandle < streamBrowserHandle
    methods
        %% constructor
        function obj = markerStreamBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
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
                defaults.gui = @streamBrowserNG;
                defaults.onscreenDisplay = true;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'gain'), defaults.gain = 0.25;end
            if ~isfield(defaults,'normalizeFlag'), defaults.normalizeFlag = false;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = false;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if defaults.windowWidth > defaults.endTime - defaults.startTime, defaults.windowWidth = defaults.endTime - defaults.startTime; end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'gui'), defaults.gui = @streamBrowserNG;end
            if ~isfield(defaults,'onscreenDisplay'), defaults.onscreenDisplay = true;end
            obj@streamBrowserHandle(dStreamObj,defaults);
            grid(obj.axesHandle,'on');
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
            delta = obj.windowWidth/2;
            if  nowCursor + delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    nowCursor - delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = nowCursor;
            elseif nowCursor + delta >= obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end)) - delta;
                if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            else newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + delta;
            end
            nowCursor = newNowCursor;
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor+obj.windowWidth/2));
            if t1==t2, return;end
            dt = length(t1:t2)/2;
            data = double(obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),obj.channelIndex));
            
            if obj.numberOfChannelsToPlot > 1
                ytick = 1.1*(0:obj.numberOfChannelsToPlot-1);
                data = data + ones(2*dt,1)*fliplr(ytick);
            else ytick = 0;
            end
            lim = [ytick(1) - 0.1 ytick(end) + 1.1];
            
            if obj.numberOfChannelsToPlot <= 1
                data = {data};
            elseif obj.numberOfChannelsToPlot == 2
                data = num2cell(data,1)';
            else data = num2cell(data,[1 obj.numberOfChannelsToPlot])';
            end
            
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                try delete( findall(obj.axesHandle,'Tag','graphics.cursorbar'));end %#ok
            end
            set(obj.gObjHandle,'XData',obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),{'YData'},data,{'Color'},obj.colorInCell);           
            
            xlim(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));
            ylim(obj.axesHandle,lim);  % static limit
            obj.axesSize.xlim = obj.streamHandle.timeStamp(obj.timeIndex([t1 t2]));
            obj.axesSize.ylim = lim;
              
            set(obj.axesHandle,'YTick',ytick);
            set(obj.axesHandle,'YTickLabel',obj.yTickLabel);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                obj.cursorHandle.gh = graphics.cursorbar(obj.gObjHandle(obj.cursorHandle.ghIndex),'Parent',obj.axesHandle);
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
        end
    end 
end