classdef cometBrowserHandle < browserHandle
    properties
        markerHandle
        textHandle
        showChannelNumber
        numberOfChannelsToPlot
        colorInCell
        floorColor
        background
        dim
        label
        tail
    end
    properties(SetObservable)
        channelIndex
        color
    end
    methods
        %% constructor
        function obj = cometBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.showChannelNumber = false;
                defaults.channels = 1;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.floorColor = [0.5294 0.6118 0.8706];
                defaults.background = [1 1 1];
                defaults.nowCursor = defaults.startTime + 2.5;
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = false;end
            if ~isfield(defaults,'channels'), defaults.channels = 1;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'floorColor'), defaults.floorColor = [0.5294 0.6118 0.8706];end
            if ~isfield(defaults,'background'), defaults.background = [1 1 1];end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + 2.5;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('channelIndex','PostSet',@cometBrowserHandle.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@cometBrowserHandle.updateTimeIndexDependencies);
            obj.addlistener('color','PostSet',@cometBrowserHandle.updateColorDependencies);
            obj.addlistener('font','PostSet',@cometBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
            obj.showChannelNumber = defaults.showChannelNumber;
            obj.tail = round(dStreamObj.samplingRate*2);
            obj.color = flipud(gray(obj.tail));
            obj.floorColor = defaults.floorColor;%[0.15 0.47 0.4];
            obj.background = defaults.background;
            obj.dim = [length(obj.streamHandle.timeStamp) obj.streamHandle.numberOfChannels];
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            %obj.timeStamp = obj.streamHandle.timeStamp(obj.timeIndex);
            obj.step = defaults.speed;       % one second
            obj.nowCursor = defaults.nowCursor;
                                                           
            obj.channelIndex = defaults.channels; 
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.showChannelNumber = obj.showChannelNumber;
            defaults.color = obj.color;
            defaults.floorColor = obj.floorColor;
            defaults.background = obj.background;
            defaults.label = obj.label;
            defaults.streamName = obj.streamHandle.name;
            defaults.browserType = 'cometBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            %obj.dcmHandle.Enable = 'off';
            box(obj.axesHandle,'off');
            
            % find now cursor index
            delta = obj.tail/obj.streamHandle.samplingRate;
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-delta/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+delta/2)));  
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor - delta/2);
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor + delta/2);
            
            obj.streamHandle.reshape([obj.dim(1) 3 obj.dim(2)/3]);
            data = squeeze(obj.streamHandle.data(obj.timeIndex(t1):obj.timeIndex(t2)-1,:,obj.channelIndex));
            obj.streamHandle.reshape([obj.dim(1) obj.dim(2)]);
            
            %if obj.numberOfChannelsToPlot==1, data = data';end
            data(isnan(data(:))) = NaN;     
            
            sceneView = get(obj.axesHandle,'view');
            
            delete(obj.markerHandle(ishandle(obj.markerHandle)));
            obj.markerHandle = [];
            cla(obj.axesHandle);
            set(obj.axesHandle,'nextplot','add')
            
            if obj.showChannelNumber
                I = squeeze(sum(data(end,:,:),2))~=0;
                try delete(obj.textHandle);end %#ok
                obj.textHandle = text(double(1.01*data(end,1,I)),double(1.01*data(end,3,I)),double(1.01*data(end,2,I)),obj.label(I),'Color',[1 0 1],'Parent',obj.axesHandle);
            end
            
            for it=1:obj.numberOfChannelsToPlot
                obj.markerHandle(it) = scatter3(data(:,1,it),data(:,3,it),data(:,2,it),'CData',obj.color,'Parent',obj.axesHandle);
            end
            
            hold(obj.axesHandle,'on')
            [x,y,z] = meshgrid(obj.streamHandle.animationParameters.limits(1,:),obj.streamHandle.animationParameters.limits(2,:),...
                obj.streamHandle.animationParameters.limits(3,1));
            surf(obj.axesHandle,double(x),double(y),double(z),'FaceColor',obj.floorColor);
            hold(obj.axesHandle,'off')
            
            xlabel(obj.axesHandle,'x','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            zlabel(obj.axesHandle,'y','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,'z','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            title(obj.axesHandle,'')
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor) ' sec']);
            
            set(obj.axesHandle,'xlim',obj.streamHandle.animationParameters.limits(1,:),'ylim',obj.streamHandle.animationParameters.limits(2,:),'zlim',...
                obj.streamHandle.animationParameters.limits(3,:),'Color',obj.background,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.axesHandle,'view',sceneView);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
            delta = obj.tail/obj.streamHandle.samplingRate;
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
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-delta)));  
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor - delta);
            t2 = t1+obj.tail;
            obj.streamHandle.reshape([obj.dim(1) 3 obj.dim(2)/3]);
            data = squeeze(obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1):obj.timeIndex(t2)-1,:,obj.channelIndex));
            obj.streamHandle.reshape([obj.dim(1) obj.dim(2)]);
            
            if obj.showChannelNumber
                I = squeeze(sum(data(end,:,:),2))~=0;
                try delete(obj.textHandle);end %#ok
                obj.textHandle = text(double(1.01*data(end,1,I)),double(1.01*data(end,3,I)),double(1.01*data(end,2,I)),obj.label(I),'Color',[1 0 1],'Parent',obj.axesHandle);
            end
            
            data(isnan(data(:))) = NaN;
            for it=1:obj.numberOfChannelsToPlot
                set(obj.markerHandle(it),'XData',data(:,1,it),'YData',data(:,3,it),'ZData',data(:,2,it));
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
        function changeChannelColor(obj,newColor,index)
            if nargin < 2, warndlg2('You must enter the index of the channel you would like change the color.');end
            ind = ismember(obj.channelIndex, index);
            if ~any(ind), errordlg2('Index exceeds the number of channels.');end
            if length(newColor) == 3
                obj.color(index,:) = ones(length(index),1)*newColor;
            end
        end
        %%
        function obj = changeSettings(obj)
             
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
            prefObj = [...
                PropertyGridField('channelIndex',1:obj.streamHandle.numberOfChannels/3,'DisplayName','Markers to show','Description','')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('showChannelNumber',obj.showChannelNumber,'DisplayName','Show marker number','Description','')...
                PropertyGridField('floorColor',obj.floorColor,'DisplayName','Floor color ','Description','')...
                PropertyGridField('background',obj.background,'DisplayName','Background color ','Description','')...
                PropertyGridField('tail',obj.tail/obj.streamHandle.samplingRate,'DisplayName','Tail','Description','Specifies the length in seconds of the trajectory represented in degraded color (as a comet''s tail)')...
                ];
            
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.showChannelNumber = val.showChannelNumber;
            obj.channelIndex = val.channelIndex;
            obj.floorColor = val.floorColor;
            obj.tail = round(val.tail*obj.streamHandle.samplingRate);
            obj.color = flipud(gray(obj.tail));
            obj.background = val.background;
            if any(obj.background < 0.4), obj.color = gray(obj.tail);end
            figure(obj.figureHandle);
            obj.createGraphicObjects(obj.nowCursor);
        end
        %%
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);%#ok
            if ~any(I), return;end
            obj.channelIndex = channelIndex;
        end
        %%
        function connectMarkers(obj)
            set(obj.markerHandle,'HitTest','on');
            try set(obj.lineHandle,'HitTest','off');end %#ok
        end
        %%
        function deleteConnection(obj)
            set(obj.markerHandle,'HitTest','off');
            try set(obj.lineHandle,'HitTest','on');end %#ok
        end
    end
    %%
    methods(Static)
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            evnt.AffectedObject.label = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            labels = cell(evnt.AffectedObject.numberOfChannelsToPlot,1);
            channels = evnt.AffectedObject.channelIndex;
            if evnt.AffectedObject.numberOfChannelsToPlot > 1
                for jt=1:evnt.AffectedObject.numberOfChannelsToPlot, labels{jt} = num2str(channels(jt));end
            else
                labels{1} = num2str(channels);
            end
            evnt.AffectedObject.label = labels;
        end
        %%
        function updateTimeIndexDependencies(~,evnt)
            if evnt.AffectedObject.timeIndex(1) ~= -1
                evnt.AffectedObject.nowCursor = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)) + 2.5;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
        %%
        function updateColorDependencies(~,evnt)
            evnt.AffectedObject.colorInCell = cell(evnt.AffectedObject.tail,1);
            for jt=1:evnt.AffectedObject.tail
                evnt.AffectedObject.colorInCell{jt} = evnt.AffectedObject.color(jt,:);
            end
            evnt.AffectedObject.colorInCell = repmat(evnt.AffectedObject.colorInCell,evnt.AffectedObject.numberOfChannelsToPlot,1);
        end
        %%
        function updateFont(~,evnt)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            
            xlabel(evnt.AffectedObject.axesHandle,'x','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            zlabel(evnt.AffectedObject.axesHandle,'y','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            ylabel(evnt.AffectedObject.axesHandle,'z','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
    end
end
