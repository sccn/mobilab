classdef DataStreamBrowser < CoreBrowser
    properties
        windowWidth = 5;
        normalizeFlag = false;
        showChannelNumber = false;
        gain = 1;
        numberOfChannelsToPlot
        yTickLabel
        colormap = 'lines';
        colorInCell
        textHandle;
        isEpoched
        dim
    end
    properties(SetObservable)
        color
    end
    methods
        %% constructor
        function obj = DataStreamBrowser(streamHandle, plotMode, master)
            if nargin < 2, plotMode = 'standalone';end
            if nargin < 3, master = -1;end
            obj.streamHandle = streamHandle;
            obj.dim = fliplr(size(streamHandle.data));
            obj.addlistener('channelIndex','PostSet',@DataStreamBrowser.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@DataStreamBrowser.updateTimeIndexDenpendencies);
            obj.addlistener('color','PostSet',@DataStreamBrowser.updateColorInCell);
            obj.mode = plotMode;
            obj.master = master;
            obj.init();
        end
        %% plot
        function init(obj)
            init@CoreBrowser(obj);
            set(obj.figureHandle,'Renderer','Painters','CloseRequestFcn',@(src, event)onClose(obj,[], event))
            set(obj.figureHandle,'name','Scroll channel activities');
            
            % find now cursor index
            [~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            [~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),obj.channelIndex);
            
            cla(obj.axesHandle);
            hold(obj.axesHandle,'on');
            obj.gObjHandle = plot(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),data);
            for it=1:obj.numberOfChannelsToPlot
                set(obj.gObjHandle(it),'color',obj.color(it,:),'userData',{obj.streamHandle, obj.channelIndex(it)});
            end
            obj.plotThisTimeStamp(obj.nowCursor);
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
            [~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            [~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            if t1==t2, return;end
            dt = length(t1:t2)/2;
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),obj.channelIndex);
            if sum(data(~isnan(data)))
                if obj.normalizeFlag
                    [data,~,sigma] = zscore(data);
                    sigma(isnan(sigma)) = 1;
                else
                    [~,mu,sigma] = zscore(data);
                    mu(isnan(mu)) = 0;
                    sigma(isnan(sigma)) = 1;
                    data = data - ones(2*dt,1)*mu;
                end
            else
                sigma = 1;
            end
            sigma(sigma == 0) = 1;
            if obj.numberOfChannelsToPlot > 1
                ytick = (1:obj.numberOfChannelsToPlot)*mean(sigma)/obj.gain;
                data = data + ones(2*dt,1)*fliplr(1:obj.numberOfChannelsToPlot)*mean(sigma)/obj.gain;
                delta = abs(diff(ytick([2 1])));
                lim = [ytick(1) - delta ytick(end) + delta];
            elseif obj.numberOfChannelsToPlot == 1 && obj.streamHandle.numberOfChannels > 1
                data = data/max([data; eps]);
                ytick = mean(data);
                mx = 1.5*max(abs(data));
                lim = [ytick - mx ytick + mx];
            else
                if sum(data(:))
                    data = data./max(abs(data));
                    ytick = min(abs([min(data(:)) max(data(:))]));
                    lim = [ytick-max(abs(data))*1.5 ytick+max(abs(data))*1.5];
                else
                    ytick = 0;
                    lim = [-1 1];
                end
            end
            if sum(data(:)) == 0, ytick = 0; lim = [-1 1];end
            
            if obj.numberOfChannelsToPlot <= 1
                data = {data};
            elseif obj.numberOfChannelsToPlot == 2
                data = num2cell(data,1)';
            else
                data = num2cell(data,[1 obj.numberOfChannelsToPlot])';
            end
            set(obj.gObjHandle,'XData',obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),{'YData'},data,{'Color'},obj.colorInCell);           
            xlim(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));
            ylim(obj.axesHandle,lim);  % static limit
            
            if obj.showEvents
                if length(obj.eventObj.latencyInFrame) ~= size(obj.eventColor,1), obj.initEventColor;end
                [~,loc1,loc2] = intersect(obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));
                Nloc1 = length(loc1);
                if ~isempty(loc1)
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    ind = obj.timeIndex(t1:t2);
                    linesHandler = line(ones(2,1)*obj.streamHandle.timeStamp(ind(loc1)),(ones(length(loc1),1)*lim)','Parent',obj.axesHandle);
                    textPos = [obj.streamHandle.timeStamp(ind(loc1))-0.5*(obj.streamHandle.timeStamp(ind(loc1))-...
                            obj.streamHandle.timeStamp(ind(loc1)-0));ones(1,Nloc1)*lim(2)*1.01];
                    try delete(obj.textHandle);end %#ok
                    set(linesHandler,{'color'},num2cell(obj.eventColor(loc2,:)',[1 3])');
                    obj.textHandle = zeros(length(loc1),1);
                    for it=1:Nloc1
                        obj.textHandle(it) = text('Position',textPos(:,it),'String',obj.eventObj.label(loc2(it)),'Color',obj.eventColor(loc2(it),:),...
                            'Parent',obj.axesHandle,'FontSize',12,'FontWeight','bold','Rotation',45);
                    end
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    hold(obj.axesHandle,'off');
                end
            end
            set(obj.axesHandle,'YTick',ytick);
            set(obj.axesHandle,'YTickLabel',obj.yTickLabel);
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
        function changeColormap(obj,newColormap)
            if nargin < 2, newColormap = '';end
            try
                if strcmpi(newColormap,'eegplot')
                    obj.color = ones(obj.numberOfChannelsToPlot,1)*[0 0 0.4];
                else 
                    obj.color = eval([newColormap '(' num2str(obj.numberOfChannelsToPlot) ')']);
                end
                obj.colormap = newColormap;
            catch
                warndlg(['Colormap ' newColormap ' is not available. We will use jet instead.'])
                obj.color = jet(obj.numberOfChannelsToPlot);
                obj.colormap = 'jet';
            end
        end
        %%
        function onClose(obj,~,~)
            delete(obj);
        end
        function delete(obj)
            delete(obj.figureHandle);
        end
        
        %%
        function obj = changeSettings(obj)
            prompt = {'Channel gain','Channels to plot','Speed','Window width','Normalize','Show channel number','Show events','Colormap'};
            defaultVal = {num2str(obj.gain), ['[' num2str(obj.channelIndex) ']'], num2str(obj.speed), num2str(obj.windowWidth), num2str(obj.normalizeFlag),...
                num2str(obj.showChannelNumber), num2str(obj.showEvents), obj.colormap};
            properties = inputdlg(prompt,'Preferences',1,defaultVal);
            
            if isempty(properties)
                return;
            end
            obj.gain = abs(str2num(properties{1}));                 %#ok
            obj.channelIndex = str2num(properties{2});              %#ok
            tmp = str2double(properties{3});
            obj.speed = interp1(1:5,1:5,tmp,'nearest','extrap');
            obj.windowWidth = abs(str2num(properties{4}));          %#ok
            obj.normalizeFlag = logical(str2num(properties{5}));    %#ok
            obj.showChannelNumber = logical(str2num(properties{6}));%#ok
            
            showEvents = str2num(properties{7});                    %#ok
            if ~showEvents
                obj.eventObj = event;
            end
            obj.changeColormap(properties{8});
            
            figure(obj.figureHandle);
            obj.init();
        end
    end
    %%
    methods(Static)
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            
            evnt.AffectedObject.yTickLabel = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            labels = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            channels = evnt.AffectedObject.channelIndex;
            
            if evnt.AffectedObject.showChannelNumber
                if evnt.AffectedObject.numberOfChannelsToPlot > 1
                    for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot), labels{jt} = num2str(channels(evnt.AffectedObject.numberOfChannelsToPlot-jt+1));end
                else
                    labels{1} = num2str(channels);
                end
            else
                if evnt.AffectedObject.numberOfChannelsToPlot > 1
                    if isempty(evnt.AffectedObject.streamHandle.label)
                        for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot), labels{jt} = num2str(evnt.AffectedObject.numberOfChannelsToPlot-jt+1);end
                    else
                        for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot), labels{jt} = evnt.AffectedObject.streamHandle.label{channels(evnt.AffectedObject.numberOfChannelsToPlot-jt+1)};end
                    end
                else
                    if ~isempty(evnt.AffectedObject.streamHandle.label)
                        labels = evnt.AffectedObject.streamHandle.label(evnt.AffectedObject.channelIndex);
                    else
                        labels{1} = '';
                    end
                end
            end
            evnt.AffectedObject.yTickLabel = labels;
            evnt.AffectedObject.changeColormap(evnt.AffectedObject.colormap);
        end
        %%
        function updateTimeIndexDenpendencies(~,evnt)
            if evnt.AffectedObject.timeIndex(1) ~= -1
                evnt.AffectedObject.nowCursor = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)) + 2.5;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
        %%
        function updateColorInCell(~,evnt)
            evnt.AffectedObject.colorInCell = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            for it=1:evnt.AffectedObject.numberOfChannelsToPlot
                evnt.AffectedObject.colorInCell{it} = evnt.AffectedObject.color(it,:);
            end
        end
    end
end