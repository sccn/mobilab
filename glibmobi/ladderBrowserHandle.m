classdef ladderBrowserHandle < browserHandle
    properties        
        gObjHandle
        windowWidth
        textHandle
        xlim
        ylim
        color
        colormap
        osdPoints
        osdText
        osdColor
        superIndex
        eventObj
        ribbonColor
        ribbonCData
        time
        leaderFollower
    end
    properties(Constant)
        numberOfChannelsToPlot = 2;
    end
    properties(SetObservable)
        channelIndex
    end
    methods
        %% constructor
        function obj = ladderBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.channelIndex = 1:dStreamObj.numberOfChannels/numel(dStreamObj.componetsToProject);
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'channelIndex'), defaults.channelIndex = 1:2;end
            if defaults.windowWidth > defaults.endTime - defaults.startTime, defaults.windowWidth = defaults.endTime - defaults.startTime; end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
                                    
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.eventObj = dStreamObj.event;
            obj.font = defaults.font;
            obj.addlistener('channelIndex','PostSet',@ladderBrowserHandle.updateChannelDependencies);
            obj.addlistener('font','PostSet',@ladderBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.cursorHandle = [];
            obj.state = false;
            obj.textHandle = [];
            obj.colormap = 'lines';
            
            obj.timeIndex = 1: size(dStreamObj,1);
            obj.time = (0:size(dStreamObj,1)-1)/dStreamObj.samplingRate;
            
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = obj.time(1) + obj.windowWidth/2;
            obj.onscreenDisplay = true; % onscreen display information (e.g. events, messages, etc)
            
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
        end
        %% plot
        function createGraphicObjects(obj,nowCursor) 
            createGraphicObjects@browserHandle(obj);
            set(obj.sliderHandle,'Max',obj.time(end));
            set(obj.sliderHandle,'Min',obj.time(1));
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(findobj(obj.figureHandle,'tag','text4'),'String',num2str(obj.time(1),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(findobj(obj.figureHandle,'tag','text5'),'String',num2str(obj.time(end),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            colorbar('peer',obj.axesHandle);
            grid(obj.axesHandle,'on')
            xlabel('Time','FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,obj.streamHandle.label{1},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            zlabel(obj.axesHandle,obj.streamHandle.label{2},'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            
            mx = max(abs(obj.ribbonCData));
            tmp = unique([linspace(-0.9*mx,0,4) linspace(0,0.9*mx,4)]);
            label = cell(length(tmp),1);
            for it=1:length(tmp), label{it} = num2str(tmp(it));end
            colorbar('YTickLabel',label,'YTickMode','manual','YTick',linspace(4,60,length(tmp)));
            obj.plotThisTimeStamp(nowCursor);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            delta = obj.windowWidth/2;
            if obj.time(obj.timeIndex(end)) &&...
                    nowCursor - delta > obj.time(obj.timeIndex(1))
                newNowCursor = nowCursor;
            elseif nowCursor + delta >= obj.time(obj.timeIndex(end))
                newNowCursor = obj.time(obj.timeIndex(end)) - delta;
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            else
                newNowCursor = obj.time(obj.timeIndex(1)) + delta;
            end
            
            % find now cursor index
            obj.nowCursor = newNowCursor;
            %[~,t1] = min(abs(obj.time(obj.timeIndex) - (obj.nowCursor-delta)));  
            %[~,t2] = min(abs(obj.time(obj.timeIndex) - (obj.nowCursor+delta)));
            t1 = binary_findClosest(obj.time(obj.timeIndex), (obj.nowCursor-delta));
            t2 = binary_findClosest(obj.time(obj.timeIndex), (obj.nowCursor+delta));
            
            time1 = obj.time(obj.timeIndex(t1:t2-1));
                        
            x = squeeze(obj.streamHandle.dataInXY(t1:t2-1,1,obj.channelIndex));
            y = squeeze(obj.streamHandle.dataInXY(t1:t2-1,2,obj.channelIndex));
            
            rColor = obj.ribbonColor(t1:t2-1,:);
            rCData = obj.ribbonCData(t1:t2-1,:);
            
            cla(obj.axesHandle);

            set(obj.axesHandle, 'YLim',obj.xlim,'ZLim',obj.ylim);
            box(obj.axesHandle,'off');
            xlim(obj.axesHandle,time1([1 end]));%#ok
            hold(obj.axesHandle,'on');
            
            
            hold(obj.axesHandle,'on');
            N = 64;
            ind = floor(linspace(1,size(time1,2),N));
            % line([time1(ind); time1(ind)],[x(ind,obj.leaderFollower(1)) x(ind,obj.leaderFollower(2))]',[y(ind,obj.leaderFollower(1)) y(ind,obj.leaderFollower(2))]');
            for it=1:N
                line([time1(ind(it)) time1(ind(it))],[x(ind(it),obj.leaderFollower(1)) x(ind(it),obj.leaderFollower(2))],[y(ind(it),...
                    obj.leaderFollower(1)) y(ind(it),obj.leaderFollower(2))],'linewidth',4,'color',rColor(ind(it),:))
                % patch(t(:,it),X(:,it),Y(:,it),'g','EdgeColor',rColor(it,:),'CData',rCData(it));
            end
            plot3(time1,x(:,obj.leaderFollower(1)),y(:,obj.leaderFollower(1)),'r','linewidth',2)
            plot3(time1,x(:,obj.leaderFollower(2)),y(:,obj.leaderFollower(2)),'b','linewidth',2);

            hold(obj.axesHandle,'off');
            
            if obj.onscreenDisplay 
                hold(obj.axesHandle,'on');
                if length(obj.eventObj.latencyInFrame) ~= size(obj.osdColor,1), obj.initOsdColor;end
                [~,loc1,loc2] = intersect(t1:t2-1,obj.eventObj.latencyInFrame);
                                
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
                        obj.osdPoints(it) = plot3(obj.axesHandle,time1(loc1(it)),dataX(it),dataY(it),'o','linewidth',2,...
                            'MarkerFaceColor','r','MarkerSize',5);
                        obj.osdText(it) = text('Position',[time1(loc1(it)) 1.05*dataX(it) 1.05*dataY(it)],...
                            'String',obj.eventObj.label(loc2(it)),'Color',obj.osdColor(loc2(it),:),...
                            'Parent',obj.axesHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
                    end
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    hold(obj.axesHandle,'off');
                else
                    delete(obj.osdText(ishandle(obj.osdText)));
                    delete(obj.osdPoints(ishandle(obj.osdPoints)));
                end
                hold(obj.axesHandle,'off');
            end
            hold(obj.axesHandle,'off');
        end
        %%
        function plotStep(obj,step)
            delta = obj.windowWidth/2;
            if obj.nowCursor+step+delta < obj.streamHandle.originalStreamObj.timeStamp(obj.timeIndex(end)) &&...
                    obj.nowCursor+step-delta > obj.streamHandle.originalStreamObj.timeStamp(obj.timeIndex(1))
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step+delta > obj.streamHandle.originalStreamObj.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.originalStreamObj.timeStamp(obj.timeIndex(end))-delta;
            else
                newNowCursor = obj.streamHandle.originalStreamObj.timeStamp(obj.timeIndex(1))+delta;
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
            if ~isa(obj.master,'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
        end
        %%
        function obj = changeSettings(obj)
            h = ribbonBrowserSettings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end%#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.speed = userData.speed;
            obj.windowWidth = userData.windowWidth;
            obj.onscreenDisplay = userData.onscreenDisplay;
            figure(obj.figureHandle);
            obj.nowCursor = obj.time(obj.timeIndex(1)) + obj.windowWidth/2;
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
            if length(evnt.AffectedObject.channelIndex) ~= evnt.AffectedObject.numberOfChannelsToPlot
                error('MoBILAB:ribbon','Ribbon Browser only shows two channels at a time.');
            end
            
            evnt.AffectedObject.superIndex = reshape(1:evnt.AffectedObject.streamHandle.numberOfChannels,...
                numel(evnt.AffectedObject.streamHandle.componetsToProject),...
                evnt.AffectedObject.streamHandle.numberOfChannels/numel(...
                evnt.AffectedObject.streamHandle.componetsToProject));
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
            evnt.AffectedObject.xlim = [-mx mx];
            evnt.AffectedObject.ylim = [-mx mx];
            evnt.AffectedObject.superIndex = evnt.AffectedObject.superIndex';
            evnt.AffectedObject.changeColormap(evnt.AffectedObject.colormap);
            
            x = squeeze(evnt.AffectedObject.streamHandle.dataInXY(evnt.AffectedObject.timeIndex,1,:));
            y = squeeze(evnt.AffectedObject.streamHandle.dataInXY(evnt.AffectedObject.timeIndex,2,:));
            
            vName = ['vel_' evnt.AffectedObject.streamHandle.name];
            velItem = evnt.AffectedObject.streamHandle.container.getItemIndexFromItemName(vName);
            vObj = evnt.AffectedObject.streamHandle.container.item{velItem};
            
            ind = strfind(evnt.AffectedObject.streamHandle.name,'p1');
            if isempty(ind), ind = [2 1];else ind = [1 2];end
            evnt.AffectedObject.leaderFollower = ind;
            
            vx = squeeze(vObj.dataInXY(evnt.AffectedObject.timeIndex,1,:));
            vy = squeeze(vObj.dataInXY(evnt.AffectedObject.timeIndex,2,:));
            
            pm = [mean(x,2) mean(y,2)];
            p1 = [x(:,ind(1)) y(:,ind(1))] - pm;
            p2 = [x(:,ind(2)) y(:,ind(2))] - pm;
            indZ = all(pm==0,2); 
            vm = diff(pm,1);
            vm(end+1,:) = vm(end,:);
            vm(indZ,:) = 0;
            
            pl = projectAB(p1,vm);
            pf = projectAB(p2,vm);
            pv = pl-pf;
             
            %ang = unwrap(atan2(y(:,1),x(:,1))) - unwrap(atan2(y(:,2),x(:,2)));
            %ang = ang*180/pi;
            %evnt.AffectedObject.ribbonCData = ang;
            evnt.AffectedObject.ribbonCData = pv;
            %mu = median(evnt.AffectedObject.ribbonCData);
            %sigma = std(evnt.AffectedObject.ribbonCData);
            %I = evnt.AffectedObject.ribbonCData > mu+3*sigma;
            %evnt.AffectedObject.ribbonCData(I) = mu+3*sigma;
            %I = evnt.AffectedObject.ribbonCData < mu-3*sigma;
            %evnt.AffectedObject.ribbonCData(I) = mu-3*sigma;
            rColor = jet(64);
            c = evnt.AffectedObject.ribbonCData;
            c = c-min(c);
            c = c*63/max(c);
            c = c+1;
            c = round(c);
            evnt.AffectedObject.ribbonColor = rColor(c,:);
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
    end
end