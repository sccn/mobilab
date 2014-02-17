classdef vectorBrowserHandle < browserHandle
    properties
        gObjHandle
        textHandle
        xlim
        ylim
        colormap
        colorInCell
        superIndex
        numberOfChannelsToPlot
        showChannelNumber
        label
    end
    properties(SetObservable)
        channelIndex
        color
    end
    methods
        %% constructor
        function obj = vectorBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.showChannelNumber = true;
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'channelIndex'), defaults.channelIndex = 1:dStreamObj.numberOfChannels/numel(dStreamObj.componetsToProject);end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = true;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('channelIndex','PostSet',@vectorBrowserHandle.updateChannelDependencies);
            obj.addlistener('color','PostSet',@vectorBrowserHandle.updateColorInCell);
            obj.addlistener('font','PostSet',@vectorBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
            obj.colormap = 'lines';
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            obj.step = defaults.step;       % half a second
            obj.nowCursor = dStreamObj.timeStamp(obj.timeIndex(1))+2.5;
            obj.showChannelNumber = defaults.showChannelNumber;
            
            obj.channelIndex = defaults.channelIndex; 
            
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.streamName = obj.streamHandle.name;
            defaults.channelIndex = obj.channelIndex;
            defaults.browserType = 'vectorBrowser';
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.axesHandle,'Color',[1 1 1]);
            
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor);
            data = obj.streamHandle.data(obj.timeIndex(t0),:);
            X = data(obj.superIndex(:,1));
            Y = data(obj.superIndex(:,2));
            % X = ones(obj.numberOfChannelsToPlot,1)*obj.xlim;
            % Y = ones(obj.numberOfChannelsToPlot,1)*obj.ylim;
            
            myCompassPlot(obj,X,Y,[obj.xlim obj.ylim]);
            set(obj.gObjHandle,{'Color'},obj.colorInCell,'LineWidth',1.5);
            
            for it=1:obj.numberOfChannelsToPlot
                if obj.showChannelNumber
                    obj.textHandle(it) = text('Position',[X(it) 0.5*Y(it)],'String',obj.label{it},'Color',obj.color(it,:),'Visible','on','Parent',obj.axesHandle);
                else
                    obj.textHandle(it) = text('Position',[X(it) 0.5*Y(it)],'String',obj.label{it},'Color',obj.color(it,:),'Visible','off''Parent',obj.axesHandle);
                end
            end
            
            view(obj.axesHandle,[0 90]);
            
            xlabel(obj.axesHandle,obj.streamHandle.label{1},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,obj.streamHandle.label{2},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            
            %obj.plotThisTimeStamp(nowCursor);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
                nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));  
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor);
            data = obj.streamHandle.data(obj.timeIndex(t0),:);
            X = data(obj.superIndex(:,1));
            Y = data(obj.superIndex(:,2));
            
            myCompassPlot(obj,X,Y);
            if obj.showChannelNumber
                set(obj.textHandle,{'Position'},num2cell([X' 0.5*Y'],2),'Visible','on');
                drawnow;
            else
                set(obj.textHandle,'Visible','off');
            end
            
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
        end
        %%
        function plotStep(obj,step)
            if obj.nowCursor+step < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    obj.nowCursor+step > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step > obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
            else
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1));
            end
            obj.plotThisTimeStamp(newNowCursor);
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
        function obj = changeSettings(obj)
            h = vectorBrowserSettings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end%#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.speed = userData.speed;
            obj.channelIndex = userData.channels;
            obj.showChannelNumber = userData.showChannelNumber;
            figure(obj.figureHandle);
            obj.createGraphicObjects(obj.nowCursor);
        end
    end
    %%
    methods(Static)
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            
            if evnt.AffectedObject.numberOfChannelsToPlot == 1
                evnt.AffectedObject.superIndex = [1 2];
            else
                evnt.AffectedObject.superIndex = reshape(1:evnt.AffectedObject.streamHandle.numberOfChannels,...
                    numel(evnt.AffectedObject.streamHandle.componetsToProject),...
                    evnt.AffectedObject.streamHandle.numberOfChannels/numel(...
                    evnt.AffectedObject.streamHandle.componetsToProject));
                evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex(:,evnt.AffectedObject.channelIndex);
            end
            alpha = [2 98];
            tmp = evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,1));
            evnt.AffectedObject.xlim = prctile(tmp(:),alpha);
            % evnt.AffectedObject.xlim = [min(min(evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,1))))...
            %     max(max(evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,1))))];
            evnt.AffectedObject.xlim = max(abs(evnt.AffectedObject.xlim));
            
            tmp = evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,2));
            evnt.AffectedObject.ylim = prctile(tmp(:),alpha);
            % evnt.AffectedObject.ylim = [min(min(evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,2))))...
            %     max(max(evnt.AffectedObject.streamHandle.data(:,evnt.AffectedObject.superIndex(:,2))))];
            evnt.AffectedObject.ylim = max(abs(evnt.AffectedObject.ylim));
            
            if evnt.AffectedObject.numberOfChannelsToPlot == 1
                evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex(:)';
            else
                 evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex';
            end
            label = evnt.AffectedObject.streamHandle.label( evnt.AffectedObject.channelIndex*2);%#ok
            evnt.AffectedObject.label = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            for it=1:evnt.AffectedObject.numberOfChannelsToPlot
                evnt.AffectedObject.label{it} = label{it}(2:end);%#ok
            end
            evnt.AffectedObject.changeColormap(evnt.AffectedObject.colormap);
            evnt.AffectedObject.textHandle = zeros(evnt.AffectedObject.numberOfChannelsToPlot,1);
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
        function updateFont(~,evnt)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            xlabel(evnt.AffectedObject.axesHandle,evnt.AffectedObject.streamHandle.label{1},'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            ylabel(evnt.AffectedObject.axesHandle,evnt.AffectedObject.streamHandle.label{2},'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
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