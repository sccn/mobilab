classdef generalizedCoordinatesBrowserHandle < browserHandle
    properties
        children
        gObjHandle
        windowWidth
        normalizeFlag
        showChannelNumber
        gain
        numberOfChannelsToPlot
        yTickLabel
        colormap
        colorInCell
        osdColor  % onscreen display color
        textHandle
        dim
        linearIndex
        coordinates
    end
    properties(SetObservable)
        channelIndex
        color
    end
    methods
        function obj = generalizedCoordinatesBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.gain = 0.25;
                defaults.normalizeFlag = false;
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels/3;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.coordinates = true(3,1);
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'gain'), defaults.gain = 0.25;end
            if ~isfield(defaults,'normalizeFlag'), defaults.normalizeFlag = false;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = false;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels/3;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'coordinates'), defaults.coordinates = true(3,1);end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            obj.uuid = defaults.uuid;
            [~,BGobj] = dStreamObj.container.viewLogicalStructure('',false);
            index = dStreamObj.container.findItem(dStreamObj.uuid);
            childNodes = unique([BGobj.getDescendants(index+1)-1 index]);
            obj.children = zeros(length(childNodes)-1,1);
            if length(obj.children) == 1, error('Cannot find the streams containing the derivatives.');end
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('channelIndex','PostSet',@generalizedCoordinatesBrowserHandle.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@generalizedCoordinatesBrowserHandle.updateTimeIndexDenpendencies);
            obj.addlistener('color','PostSet',@generalizedCoordinatesBrowserHandle.updateColorInCell);
            obj.addlistener('font','PostSet',@generalizedCoordinatesBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
            obj.normalizeFlag = defaults.normalizeFlag;
            obj.showChannelNumber = false;
            obj.colormap = 'lines';
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2;
            %obj.timeStamp = obj.streamHandle.timeStamp(obj.timeIndex);
            obj.gain = defaults.gain;
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.onscreenDisplay = true; % onscreen display information (e.g. events, messages, etc)
            obj.coordinates = defaults.coordinates;
            obj.channelIndex = defaults.channels;
            
            tmpColor = lines(length(obj.streamHandle.event.uniqueLabel));
            obj.osdColor = zeros(length(obj.streamHandle.event.latencyInFrame),3);
            for it=1:length(obj.streamHandle.event.uniqueLabel)
                loc = ismember(obj.streamHandle.event.label,obj.streamHandle.event.uniqueLabel(it));
                obj.osdColor(loc,1) = tmpColor(it,1);
                obj.osdColor(loc,2) = tmpColor(it,2);
                obj.osdColor(loc,3) = tmpColor(it,3);
            end
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.windowWidth = obj.windowWidth;
            defaults.normalizeFlag = obj.normalizeFlag;
            defaults.showChannelNumber = obj.showChannelNumber;
            defaults.gain = obj.gain;
            defaults.colormap = obj.colormap;
            defaults.colormap = obj.colormap;
            defaults.yTickLabel = obj.yTickLabel;
            defaults.osdColor = obj.osdColor;
            defaults.streamName = obj.streamHandle.name;
            defaults.browserType = 'generalizedCoordinatesBrowser';
            defaults.channels = obj.channelIndex;
            defaults.coordinates = obj.coordinates;
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            view(obj.axesHandle,[0 90]);
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2))); 
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor+obj.windowWidth/2)); 
            Nt = length(t1:t2);           
            Nc = length(obj.children);
            data = zeros(Nt,(Nc+1)*obj.numberOfChannelsToPlot*sum(obj.coordinates));
            cla(obj.axesHandle);
            hold(obj.axesHandle,'on');
            obj.gObjHandle = plot(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),data);
            obj.cursorHandle = plot(obj.axesHandle,ones(4,1)*obj.nowCursor,linspace(0,10,4),'LineWidth',2,'Color','r');
            hold(obj.axesHandle,'off');
            set(obj.figureHandle,'name',['MoBILAB generalizedCoordinatesBrowser: ' obj.streamHandle.name]);
            % obj.plotThisTimeStamp(nowCursor);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
            delta = obj.windowWidth/2;
            if  nowCursor + delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    nowCursor - delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = nowCursor;
            elseif nowCursor + delta > obj.streamHandle.timeStamp(obj.timeIndex(end))
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
            Nt = length(t1:t2);
            dt = Nt/2;
            
            Nc = length(obj.children);
            
            data = zeros(Nt,Nc+1,obj.numberOfChannelsToPlot*obj.dim(3));
            data(:,1,:) = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),obj.linearIndex);
            for it=1:Nc
                data(:,it+1,:) = obj.streamHandle.container.item{obj.children(it)}.data(obj.timeIndex(t1:t2),obj.linearIndex);
            end
            data = reshape(data,[Nt (Nc+1)*obj.numberOfChannelsToPlot*obj.dim(3)]);
            
            if obj.normalizeFlag
                [data,~,sigma] = zscore(data);
            else
                [~,mu,sigma] = zscore(data);
                data = data - ones(2*dt,1)*mu;
            end
            sigma(sigma == 0) = 1;

            ytick = (1:obj.numberOfChannelsToPlot*(Nc+1)*obj.dim(3))*mean(sigma)/obj.gain;
            data = data + ones(2*dt,1)*fliplr(1:obj.numberOfChannelsToPlot*obj.dim(3)*(Nc+1))*mean(sigma)/obj.gain;
            delta = abs(diff(ytick([2 1])));
            lim = [ytick(1) - delta ytick(end) + delta];
            if sum(data(:)) == 0, ytick = 0; lim = [-1 1];end
            
            if strcmp(obj.dcmHandle.Enable,'on')
                set(obj.cursorHandle,'XData',ones(4,1)*obj.nowCursor,'YData',linspace(lim(1),lim(2),4),'Visible','on');
            else
                set(obj.cursorHandle,'Visible','off');
            end
            obj.dcmHandle.removeAllDataCursors;
            
            data = num2cell(data,[1 length(obj.gObjHandle)]);
            set(obj.gObjHandle(:),'XData',obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),{'YData'},data',...
                {'color'},obj.colorInCell);
                         
            xlim(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));
            ylim(obj.axesHandle,lim);  % static limit
            
            if obj.onscreenDisplay
                [~,loc1,loc2] = intersect(obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame));
                                
                if ~isempty(loc1)
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    ind = obj.timeIndex(t1:t2);
                    linesHandler = line(ones(2,1)*obj.streamHandle.timeStamp(ind(loc1)),(ones(length(loc1),1)*lim)','Parent',obj.axesHandle);
                    try delete(obj.textHandle);end %#ok
                    obj.textHandle = zeros(length(loc1),1);
                    for it=1:length(loc1)
                        set(linesHandler(it),'color',obj.osdColor(loc2(it),:));
                        obj.textHandle(it) = text('Position',[obj.streamHandle.timeStamp(ind(loc1(it)))-0.5*(obj.streamHandle.timeStamp(ind(loc1(it)))-...
                            obj.streamHandle.timeStamp(ind(loc1(it))-1)),lim(2)+lim(2)/28],'String',obj.streamHandle.event.label(loc2(it)),'Color',...
                            obj.osdColor(loc2(it),:),'Parent',obj.axesHandle);
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
        function delete(obj)
            try delete(obj.figureHandle);end %#ok
            if ~strcmp(class(obj.master),'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
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
        function obj = changeSettings(obj)
            h = genCoordBrowserSettings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end%#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.gain = userData.gain;
            obj.speed = userData.speed;
            obj.windowWidth = userData.windowWidth;
            obj.normalizeFlag = userData.normalizeFlag;
            obj.showChannelNumber = userData.showChannelNumber;
            obj.coordinates = ~userData.coordinates;
            obj.channelIndex = userData.channels;
            obj.onscreenDisplay = userData.onscreenDisplay;
            obj.changeColormap(userData.colormap);
            if isfield(userData.newColor,'color') && isfield(userData.newColor,'channel')
                obj.changeChannelColor(userData.newColor.color,userData.newColor.channel);
            end
            
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
        %% color functions
        function changeChannelColor(obj,newColor,index)
            if nargin < 2, warndlg2('You must enter the index of the channel you would like change the color.');end
            ind = ismember(obj.channelIndex, index);
            if ~any(ind), errordlg2('Index exceeds the number of channels.');end
            if length(newColor) == 3
                obj.color(ind,:) = ones(sum(ind),1)*newColor;
            end
        end
        %%
        function changeColormap(obj,newColormap)
            if nargin < 2, newColormap = '';end
            switch lower(newColormap)
                case 'jet'
                    obj.color = jet(obj.numberOfChannelsToPlot);
                    obj.colormap = 'jet';
                case 'hsv'
                    obj.color = hsv(obj.numberOfChannelsToPlot);
                    obj.colormap = 'hsv';
                case 'hot'
                    obj.color = hot(obj.numberOfChannelsToPlot);
                    obj.colormap = 'hot';
                case 'cool'
                    obj.color = cool(obj.numberOfChannelsToPlot);
                    obj.colormap = 'cool';
                case 'spring'
                    obj.color = spring(obj.numberOfChannelsToPlot);
                    obj.colormap = 'spring';
                case 'summer'
                    obj.color = summer(obj.numberOfChannelsToPlot);
                    obj.colormap = 'summer';
                case 'autumn'
                    obj.color = autumn(obj.numberOfChannelsToPlot);
                    obj.colormap = 'autumn';
                case 'winter'
                    obj.color = winter(obj.numberOfChannelsToPlot);
                    obj.colormap = 'winter';
                case 'gray'
                    obj.color = gray(obj.numberOfChannelsToPlot);
                    obj.colormap = 'gray';
                case 'bone'
                    obj.color = bone(obj.numberOfChannelsToPlot);
                    obj.colormap = 'bone';
                case 'copper'
                    obj.color = copper(obj.numberOfChannelsToPlot);
                    obj.colormap = 'copper';
                case 'pink'
                    obj.color = pink(obj.numberOfChannelsToPlot);
                    obj.colormap = 'pink';
                case 'lines'
                    obj.color = lines(obj.numberOfChannelsToPlot);
                    obj.colormap = 'lines';
                case 'eegplot'
                    obj.color = ones(obj.numberOfChannelsToPlot,1)*[0 0 0.4];
                    obj.colormap = 'eegplot';
                case 'custom'
                    obj.colormap = 'custom';
                otherwise
                    warndlg2('This colormap is not available. See the options for ''colormap'' in MATLAB documentation.')
            end
        end
    end
    %%
    methods(Static)
        %%
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            evnt.AffectedObject.dim = [length(evnt.AffectedObject.streamHandle.timeStamp) evnt.AffectedObject.streamHandle.numberOfChannels...
                sum(evnt.AffectedObject.coordinates)];
            
            evnt.AffectedObject.yTickLabel = cell(evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3),...
                length(evnt.AffectedObject.children)+1);
            
            labels = cell(evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3),1);
            
            index = reshape(1:evnt.AffectedObject.streamHandle.numberOfChannels,3,evnt.AffectedObject.streamHandle.numberOfChannels/3);
            evnt.AffectedObject.linearIndex = index(evnt.AffectedObject.coordinates,evnt.AffectedObject.channelIndex);
            evnt.AffectedObject.linearIndex = evnt.AffectedObject.linearIndex(:);
                
            if evnt.AffectedObject.showChannelNumber
                for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3)),
                    labels{jt} = num2str(evnt.AffectedObject.linearIndex(evnt.AffectedObject.numberOfChannelsToPlot*...
                        evnt.AffectedObject.dim(3)-jt+1));
                end
            else
                for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3))
                    labels{jt} = evnt.AffectedObject.streamHandle.label{evnt.AffectedObject.linearIndex(evnt.AffectedObject.numberOfChannelsToPlot*...
                        evnt.AffectedObject.dim(3)-jt+1)};
                end
            end
            evnt.AffectedObject.yTickLabel(:,1) = labels;
            
            for k=1:length(evnt.AffectedObject.children)
                devChar = repmat('''',1,k);
                if evnt.AffectedObject.showChannelNumber
                    for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3))
                        labels{jt} = [num2str(evnt.AffectedObject.linearIndex(evnt.AffectedObject.numberOfChannelsToPlot*...
                            evnt.AffectedObject.dim(3)-jt+1)) devChar];
                    end
                else
                    for jt=fliplr(1:evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3))
                        labels{jt} = [evnt.AffectedObject.streamHandle.label{evnt.AffectedObject.linearIndex(...
                            evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3)-jt+1)} devChar];
                    end
                end
                evnt.AffectedObject.yTickLabel(:,k+1) = labels;
            end
            evnt.AffectedObject.yTickLabel = fliplr(evnt.AffectedObject.yTickLabel)';
            evnt.AffectedObject.yTickLabel = evnt.AffectedObject.yTickLabel(:);
            
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
            Nc = length(evnt.AffectedObject.children);
            NgHand = (Nc+1)*evnt.AffectedObject.numberOfChannelsToPlot*evnt.AffectedObject.dim(3);
            evnt.AffectedObject.colorInCell = num2cell(kron(evnt.AffectedObject.color,ones((Nc+1)*evnt.AffectedObject.dim(3),1))',[NgHand 1])';
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