classdef projectionBrowserHandle < browserHandle
    properties
        gObjHandle
        windowWidth
        textHandle
        osdPoints
        osdText
        xlim
        ylim
        colormap
        color
        superIndex
        numberOfChannelsToPlot
        osdColor
        %-
        colorCodeKinematicFlag
        showVectorsFlag
        unitColorVectors
        colorDotObj
        arrowObj
        knormColor
        magnitudeOrCurvature
        arrowDt
        eventObj
        axOriginalSize
        %-
    end
    properties(SetObservable)
        channelIndex
        colorObjIndex
    end
    methods
        %% constructor
        function obj = projectionBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.channelIndex = 1:dStreamObj.numberOfChannels/2;
                defaults.font = struct('size',12,'weight','normal');
                defaults.magnitudeOrCurvature = 1;
                defaults.colorCodeKinematicFlag = false;
                defaults.showVectorsFlag = false;
                defaults.unitColorVectors = [];
                defaults.colorDotObj = [];
                defaults.colorObjIndex = 0;
                defaults.arrowObj = [];
                defaults.knormColor = 1;
                defaults.arrowDt = 0.75*dStreamObj.samplingRate;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'channelIndex'), defaults.channelIndex = 1:dStreamObj.numberOfChannels/2;end
            if defaults.windowWidth > defaults.endTime - defaults.startTime, defaults.windowWidth = defaults.endTime - defaults.startTime; end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'colorCodeKinematicFlag'), defaults.colorCodeKinematicFlag = false;end
            if ~isfield(defaults,'showVectorsFlag'), defaults.showVectorsFlag = false;end
            if ~isfield(defaults,'colorObjIndex'), defaults.colorObjIndex = 1;end
            if ~isfield(defaults,'magnitudeOrCurvature'), defaults.magnitudeOrCurvature = true;end
            if ~isfield(defaults,'arrowDt'), defaults.arrowDt = 0.75*dStreamObj.samplingRate;end
            
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.eventObj = dStreamObj.event;
            obj.font = defaults.font;
            obj.addlistener('channelIndex','PostSet',@projectionBrowserHandle.updateChannelDependencies);
            obj.addlistener('font','PostSet',@projectionBrowserHandle.updateFont);
            obj.addlistener('colorObjIndex','PostSet',@projectionBrowserHandle.updateCodeKinematic);
            
            obj.speed = defaults.speed;
            obj.cursorHandle = [];
            obj.state = false;
            obj.textHandle = [];
            obj.colormap = 'lines';
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2-1; 
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = dStreamObj.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
            obj.onscreenDisplay = true; % onscreen display information (e.g. events, messages, etc)
            
            obj.magnitudeOrCurvature = defaults.magnitudeOrCurvature;
            obj.colorCodeKinematicFlag = defaults.colorCodeKinematicFlag;
            obj.showVectorsFlag = defaults.showVectorsFlag;
            obj.colorObjIndex = defaults.colorObjIndex;
            obj.knormColor = 1;
            obj.arrowDt = defaults.arrowDt;
            obj.channelIndex = defaults.channelIndex; 
            
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = streamBrowserNG(obj);
        end
        %
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
            defaults.streamName = obj.streamHandle.name;
            defaults.channelIndex = obj.channelIndex;
            defaults.browserType = 'projectionBrowser';
            defaults.magnitudeOrCurvature = obj.magnitudeOrCurvature;
            defaults.colorCodeKinematicFlag = obj.colorCodeKinematicFlag;
            defaults.showVectorsFlag = obj.showVectorsFlag;
            defaults.colorObjIndex = obj.colorObjIndex;
            defaults.knormColor = obj.knormColor;
            defaults.arrowDt = obj.arrowDt;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            delta = obj.windowWidth/2;
            createGraphicObjects@browserHandle(obj);
            if isempty(obj.axOriginalSize), obj.axOriginalSize = get(obj.axesHandle,'position');end
            view(obj.axesHandle,[38 36]);
            set(obj.cursorHandle.tb,'State','on');
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth)));
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth));
            t0 = fix(delta*obj.streamHandle.samplingRate);
            t2 = t1+2*t0-1;
            time = obj.streamHandle.timeStamp((t1:t2-1));
            dataX = obj.streamHandle.mmfObj.Data.x(t1:t2-1,obj.superIndex(:,1));
            dataY = obj.streamHandle.mmfObj.Data.x(t1:t2-1,obj.superIndex(:,2));
            
            cla(obj.axesHandle);
            set(obj.axesHandle, 'YLim',obj.xlim,'ZLim',obj.ylim);
            % set(obj.axesHandle2,'XLim',obj.xlim,'YLim',obj.ylim);
            box(obj.axesHandle,'off');
            xlim(obj.axesHandle,time([1 end]));%#ok
            hold(obj.axesHandle,'on');
            % hold(obj.axesHandle2,'on');
            obj.gObjHandle = zeros(obj.numberOfChannelsToPlot,1);
            obj.cursorHandle.gh = zeros(obj.numberOfChannelsToPlot,1);
                        
            hold(obj.axesHandle,'on');
            for it=1:obj.numberOfChannelsToPlot
                obj.gObjHandle(it) = plot3(obj.axesHandle,time',dataX(:,it),dataY(:,it),'Parent',obj.axesHandle,'Color',obj.color(it,:));
                obj.cursorHandle.gh(it) = plot3(obj.axesHandle,obj.nowCursor,dataX(t0,it),dataY(t0,it),...
                'o','linewidth',2,'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','r');
            end
            hold(obj.axesHandle,'off');
            
            if obj.colorCodeKinematicFlag
                hold(obj.axesHandle,'on');
                delete(obj.colorDotObj(ishandle(obj.colorDotObj)));
                obj.colorDotObj = zeros(obj.numberOfChannelsToPlot,1);
                colorCode = obj.unitColorVectors(t1:t2-1,:);
                for it=1:obj.numberOfChannelsToPlot, obj.colorDotObj(it) = scatter3(obj.axesHandle,time',dataX(:,it),dataY(:,it),'filled','CData',colorCode(:,it),'marker','o');end
                if ~isempty(strfind(lower(obj.streamHandle.container.item{obj.colorObjIndex}.name),'vel'))
                    Title = 'Velocity (m/s)';
                elseif ~isempty(strfind(lower(obj.streamHandle.container.item{obj.colorObjIndex}.name),'acc'))
                    Title = 'Acceleration (m/s^2)';
                elseif ~isempty(strfind(lower(obj.streamHandle.container.item{obj.colorObjIndex}.name),'jerk'))
                    Title = 'Jerk (m/s^3)';
                else Title = '';
                end
                set(obj.axesHandle,'position',[obj.axOriginalSize(1:2) 0.95*obj.axOriginalSize(3) obj.axOriginalSize(4)]);
                colorbar('peer',obj.axesHandle);
                title(obj.axesHandle,Title,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
                hold(obj.axesHandle,'off');
            end
            
            if obj.onscreenDisplay 
                hold(obj.axesHandle,'on');
                if length(obj.eventObj.latencyInFrame) ~= size(obj.osdColor,1), obj.initOsdColor;end
                [~,loc1,loc2] = intersect(time,obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));                
                if ~isempty(loc1) 
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                    dataX = obj.streamHandle.data(obj.eventObj.latencyInFrame(loc2),obj.superIndex(1,1));
                    dataY = obj.streamHandle.data(obj.eventObj.latencyInFrame(loc2),obj.superIndex(1,2));
                    
                    delete(obj.osdText(ishandle(obj.osdText)));
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                    obj.osdPoints = [];
                    for it=1:length(loc1)
                        obj.osdPoints(it) = plot3(obj.axesHandle,time(loc1(it)),dataX(it),dataY(it),'o','linewidth',2,...
                            'MarkerFaceColor','r','MarkerSize',5);
                        obj.osdText(it) = text('Position',[time(loc1(it)) 1.05*dataX(it) 1.05*dataY(it)],...
                            'String',obj.eventObj.label(loc2(it)),'Color',obj.osdColor(loc2(it),:),...
                            'Parent',obj.axesHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
                    end
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    hold(obj.axesHandle,'off');
                    if obj.colorCodeKinematicFlag && obj.showVectorsFlag, showThisVectors = loc2;end
                else
                    delete(obj.osdText(ishandle(obj.osdText)));
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                    showThisVectors = [];
                end
                hold(obj.axesHandle,'off');
            end
            
            if obj.showVectorsFlag && obj.colorCodeKinematicFlag
                hold(obj.axesHandle,'on');
                [~,loc1,loc2] = intersect(time,obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame)); 
                if isempty(loc1), loc1 = unique(round(0.1*length(time):obj.arrowDt:0.9*length(time)));end
                obj.arrowObj = zeros(length(loc2),obj.numberOfChannelsToPlot);
                ind = t1:t2-1;
                for it=1:obj.numberOfChannelsToPlot
                    v = obj.streamHandle.container.item{obj.colorObjIndex}.mmfObj.Data.x(ind(loc1),obj.superIndex(it,:));
                    v = bsxfun(@rdivide,v,4*sqrt(sum(v.^2,2))+eps);
                    % v(:,1) = v(:,1).*obj.knormColor(ind(loc1),it);
                    % v(:,2) = v(:,2).*obj.knormColor(ind(loc1),it);
                    obj.arrowObj(:,it) = my_arrow([time(loc1)' dataX{it}(loc1) dataY{it}(loc1)],[time(loc1)' ([dataX{it}(loc1) dataY{it}(loc1)] + v)],...
                        'Length',3,'parent',obj.axesHandle);
                end
                hold(obj.axesHandle,'off');
            end
            
            hold(obj.axesHandle,'off');
            % hold(obj.axesHandle2,'off');
                                                
            grid(obj.axesHandle,'on')
            xlabel('Time','FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,obj.streamHandle.label{1},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            zlabel(obj.axesHandle,obj.streamHandle.label{2},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            % xlabel(obj.axesHandle2,obj.streamHandle.label{1});
            % ylabel(obj.axesHandle2,obj.streamHandle.label{2});
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
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-delta)));  
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-delta));
            t0 = fix(delta*obj.streamHandle.samplingRate);
            t2 = t1+2*t0-1;
            if t2-1 > size(obj.streamHandle,1), t2 = size(obj.streamHandle)+1;end
            time = obj.streamHandle.timeStamp((t1:t2-1));
            
            dataX = obj.streamHandle.mmfObj.Data.x(t1:t2-1,obj.superIndex(:,1));
            dataY = obj.streamHandle.mmfObj.Data.x(t1:t2-1,obj.superIndex(:,2));   
            
            cursorDataX = num2cell(dataX(t0,:),1);
            cursorDataY = num2cell(dataY(t0,:),1);
            
            dataX = num2cell(dataX,1);
            dataY = num2cell(dataY,1);
        
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                set(obj.cursorHandle.gh,'XData',time(t0),{'YData'},cursorDataX',{'ZData'},cursorDataY','Visible','on');
            else set(obj.cursorHandle.gh,'Visible','off');
            end
            set(obj.gObjHandle,'XData',time',{'YData'},dataX',{'ZData'},dataY');
            xlim(obj.axesHandle,time([1 end])); %#ok
           
            if obj.colorCodeKinematicFlag
                colorCode = obj.unitColorVectors(t1:t2-1,:);
                for it=1:obj.numberOfChannelsToPlot, set(obj.colorDotObj(it),'XData',time','YData',dataX{it},'ZData',dataY{it},'CData',colorCode(:,it));end
            end
            
            if obj.onscreenDisplay
                if length(obj.eventObj.latencyInFrame) ~= size(obj.osdColor,1), obj.initOsdColor;end
                [~,loc1,loc2] = intersect(time,obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));
                
                if ~isempty(loc1)
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                    dataXe = obj.streamHandle.data(obj.eventObj.latencyInFrame(loc2),obj.superIndex(1,1));
                    dataYe = obj.streamHandle.data(obj.eventObj.latencyInFrame(loc2),obj.superIndex(1,2));
                    %obj.osdPoints = plot3(time(loc1)'*ones(1,obj.numberOfChannelsToPlot),dataX,dataY,'x','linewidth',2,'Parent',obj.axesHandle);
                    delete(obj.osdText(ishandle(obj.osdText)));
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                    obj.osdPoints = [];
                    
                    for it=1:length(loc1)
                        obj.osdPoints(it) = plot3(obj.axesHandle,time(loc1(it)),dataXe(it),dataYe(it),'o','linewidth',2,...
                            'MarkerFaceColor','r','MarkerSize',5);
                        obj.osdText(it) = text('Position',[time(loc1(it)) 1.05*dataXe(it) 1.05*dataYe(it)],...
                            'String',obj.eventObj.label(loc2(it)),'Color',obj.osdColor(loc2(it),:),...
                            'Parent',obj.axesHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
                    end
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    hold(obj.axesHandle,'off');
                else 
                    delete(obj.osdText(ishandle(obj.osdText)));
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                end
            end
            
            if obj.showVectorsFlag && obj.colorCodeKinematicFlag
                hold(obj.axesHandle,'on');
                [~,loc1] = intersect(time,obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));
                if isempty(loc1), loc1 = unique(round(0.1*length(time):obj.arrowDt:0.9*length(time)));end
                try delete(obj.arrowObj);end;%#ok
                obj.arrowObj = zeros(length(loc1),obj.numberOfChannelsToPlot);
                ind = t1:t2-1;
                for it=1:obj.numberOfChannelsToPlot
                    v = obj.streamHandle.container.item{obj.colorObjIndex}.mmfObj.Data.x(ind(loc1),obj.superIndex(it,:));
                    v = bsxfun(@rdivide,v,4*sqrt(sum(v.^2,2))+eps);
                    obj.arrowObj(:,it) = my_arrow([time(loc1)' dataX{it}(loc1) dataY{it}(loc1)],[time(loc1)' ([dataX{it}(loc1) dataY{it}(loc1)] + v)],...
                        'Length',3,'parent',obj.axesHandle);
                end
                hold(obj.axesHandle,'off');
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
                case 'jet', 
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
            
            descendants = obj.streamHandle.container.getDescendants(obj.streamHandle.container.findItem(obj.streamHandle.uuid));
            streamName = cell(length(descendants)+2,1);
            streamName{1} = 'none';
            for it=1:length(descendants), streamName{it+1} =  obj.streamHandle.container.item{descendants(it)}.name;end
            streamName{end} = 'curvature';
            
            if ~isempty(obj.unitColorVectors)
                tmpStreamName = char(streamName([false obj.colorObjIndex == descendants false]));
            else tmpStreamName = streamName{1};
            end
            prefObj = [...
                PropertyGridField('channels',obj.channelIndex,'DisplayName','Channels to plot','Description','This field accept matlab code returning a subset of channels, for instance use: ''setdiff(1:10,[3 5 7])'' to plot channels from 1 to 10 excepting 3, 5, and 7.')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                PropertyGridField('onscreenDisplay',obj.onscreenDisplay,'DisplayName','Show events on top of the trajectories','Description','')...
                PropertyGridField('labels', obj.streamHandle.event.uniqueLabel,'DisplayName','Show only a subset of events','Description','')...
                PropertyGridField('streamName',tmpStreamName,'Type',PropertyType('char','row',streamName'),'DisplayName','Color trajectory with derived measures','Description','Color trajectory with derived measures')...
                PropertyGridField('showVectorsFlag',obj.showVectorsFlag,'DisplayName','Show direction','Description','')...
                PropertyGridField('xlim',obj.xlim,'DisplayName','xlim','Description','Limits of axis x.')...
                PropertyGridField('ylim',obj.ylim,'DisplayName','xlim','Description','Limits of axis y.')];
            
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
            if ~any(ismember(1:obj.streamHandle.numberOfChannels/2,val.channels))
                warning('MoBILAB:noChannel','Error indexing channels.');
            else obj.channelIndex = val.channels;
            end
            obj.xlim = val.xlim;
            obj.ylim = val.ylim;
            obj.onscreenDisplay = val.onscreenDisplay;
            obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
            obj.colorCodeKinematicFlag = ~strcmp(val.streamName,streamName{1});
            obj.showVectorsFlag = val.showVectorsFlag;
            obj.magnitudeOrCurvature = isempty(strfind(val.streamName,'curvature'));
            if obj.colorCodeKinematicFlag
                ind = ismember(streamName(2:end-1),val.streamName);
                obj.colorObjIndex = descendants(ind);
            else obj.unitColorVectors = [];
            end
            figure(obj.figureHandle);
            obj.createGraphicObjects(obj.nowCursor);
            if isa(obj.master,'browserHandleList')
                obj.master.bound = max([obj.master.bound obj.windowWidth]);
                obj.master.nowCursor = obj.master.startTime + obj.windowWidth/2;
                obj.master.plotThisTimeStamp(obj.master.nowCursor);
            end
        end
    end
    %%
    methods(Static)
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            evnt.AffectedObject.superIndex = reshape(1:evnt.AffectedObject.streamHandle.numberOfChannels,2,...
                evnt.AffectedObject.streamHandle.numberOfChannels/2);
            if evnt.AffectedObject.numberOfChannelsToPlot == 1
                evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex(:,evnt.AffectedObject.channelIndex)';
            else
                evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex(:,evnt.AffectedObject.channelIndex);
            end
            dim = size(evnt.AffectedObject.streamHandle,1);
            I = round(0.25*dim):round(0.75*dim);
            mx = [min(evnt.AffectedObject.streamHandle.data(I,evnt.AffectedObject.superIndex(:,1))) max(evnt.AffectedObject.streamHandle.data(I,evnt.AffectedObject.superIndex(:,1)))];
            my = [min(evnt.AffectedObject.streamHandle.data(I,evnt.AffectedObject.superIndex(:,2))) max(evnt.AffectedObject.streamHandle.data(I,evnt.AffectedObject.superIndex(:,2)))];
            mx = max(abs([ mx my]));
            evnt.AffectedObject.xlim = 1.5*[-mx mx];
            evnt.AffectedObject.ylim = 1.5*[-mx mx];            
            if evnt.AffectedObject.numberOfChannelsToPlot == 1, 
                evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex(:)';
            else
                 evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex';
            end
            evnt.AffectedObject.changeColormap(evnt.AffectedObject.colormap);
            evnt.AffectedObject.colorObjIndex = evnt.AffectedObject.colorObjIndex;
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
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            xlabel(evnt.AffectedObject.axesHandle,'Time','FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            ylabel(evnt.AffectedObject.axesHandle,evnt.AffectedObject.streamHandle.label{1},'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            zlabel(evnt.AffectedObject.axesHandle,evnt.AffectedObject.streamHandle.label{2},'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
        %%
        function updateCodeKinematic(~,evnt)
            evnt.AffectedObject.unitColorVectors = zeros(length(evnt.AffectedObject.streamHandle.timeStamp),evnt.AffectedObject.numberOfChannelsToPlot);
            evnt.AffectedObject.knormColor = evnt.AffectedObject.unitColorVectors;
            if all([evnt.AffectedObject.colorCodeKinematicFlag evnt.AffectedObject.colorObjIndex])
                for it=1:evnt.AffectedObject.numberOfChannelsToPlot
                    if evnt.AffectedObject.magnitudeOrCurvature
                        evnt.AffectedObject.unitColorVectors(:,it) = evnt.AffectedObject.streamHandle.container.item{evnt.AffectedObject.colorObjIndex}.magnitude(:,evnt.AffectedObject.channelIndex(it));
                    else
                        evnt.AffectedObject.unitColorVectors(:,it) = evnt.AffectedObject.streamHandle.curvature(:,evnt.AffectedObject.channelIndex(it));
                    end
                    evnt.AffectedObject.unitColorVectors(:,it) = evnt.AffectedObject.unitColorVectors(:,it).*(1-any(evnt.AffectedObject.streamHandle.container.item{evnt.AffectedObject.colorObjIndex}.artifactMask,2));
                    evnt.AffectedObject.knormColor(:,it) = 40./(evnt.AffectedObject.unitColorVectors(:,it)+eps);
                    evnt.AffectedObject.knormColor(any(evnt.AffectedObject.streamHandle.container.item{evnt.AffectedObject.colorObjIndex}.artifactMask,2),it) = eps;
                end
            else
                evnt.AffectedObject.unitColorVectors = [];
            end
        end
    end
end