classdef mocapBrowserHandle < browserHandle
    properties
        markerHandle
        textHandle
        lineHandle
        showChannelNumber
        numberOfChannelsToPlot
        color
        lineColor
        lineWidth
        floorColor
        background
        dim
        label
        connectivity
        firstNode
        firstNodeObject
        passCursor
        beep
        roomSize
        dataXYZ = [];
    end
    properties(SetObservable)
        channelIndex
    end
    methods
        %% constructor
        function obj = mocapBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels/3;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.floorColor = [0.5294 0.6118 0.8706];
                defaults.background = [0 0 0.3059];
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = 0;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels/3;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'floorColor'), defaults.floorColor = [0.5294 0.6118 0.8706];end
            if ~isfield(defaults,'background'), defaults.background = [0 0 0.3059];end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            obj.uuid = defaults.uuid; 
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('channelIndex','PostSet',@mocapBrowserHandle.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@mocapBrowserHandle.updateTimeIndexDependencies);
            obj.addlistener('font','PostSet',@mocapBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
            obj.showChannelNumber = defaults.showChannelNumber;
            obj.lineColor   = 'y'; % lines in yellow
            obj.lineWidth   = 2;
            obj.floorColor = defaults.floorColor;%[0.15 0.47 0.4];  %[0.42 0.92 0.70])
            obj.background = defaults.background;%[0 0 0];
            obj.dim = [length(obj.streamHandle.timeStamp) obj.streamHandle.numberOfChannels];
            
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            obj.step = 1;       % one second
            obj.nowCursor = dStreamObj.timeStamp(obj.timeIndex(1)) + 2.5;
            obj.passCursor = -1;
            obj.onscreenDisplay = true; % onscreen display information (e.g. events, messages, etc)
            tmpBeep = 1*sin(2*pi*1500*(linspace(0,1,10000)));
            obj.beep = audioplayer(tmpBeep(1:1000),10000);
            
            if isa(obj.streamHandle,'mocap')
                if isempty(obj.streamHandle.animationParameters.limits), obj.streamHandle.container.findSpaceBoundary;end
                obj.roomSize.x = obj.streamHandle.animationParameters.limits(1,:);
                obj.roomSize.y = obj.streamHandle.animationParameters.limits(2,:);
                obj.roomSize.z = obj.streamHandle.animationParameters.limits(3,:);
            else
                index = dStreamObj.container.getItemIndexFromItemClass('mocap');
                mObj = dStreamObj.container.item{index(end)};
                if isempty(mObj.animationParameters.limits), mObj.container.findSpaceBoundary;end
                obj.roomSize.x = mObj.animationParameters.limits(1,:);
                obj.roomSize.y = mObj.animationParameters.limits(2,:);
                obj.roomSize.z = mObj.animationParameters.limits(3,:);
            end
            
            if ~isfield(defaults,'connectivity')
                obj.connectivity = zeros(dStreamObj.numberOfChannels/3);
                if isa(obj.streamHandle,'mocap') && isfield(obj.streamHandle.animationParameters,'conn') && ~isempty(obj.streamHandle.animationParameters.conn)
                    for it=1:size(dStreamObj.animationParameters.conn,1)
                        obj.connectivity(obj.streamHandle.animationParameters.conn(it,1),obj.streamHandle.animationParameters.conn(it,2)) = 1;
                        obj.connectivity(obj.streamHandle.animationParameters.conn(it,2),obj.streamHandle.animationParameters.conn(it,1)) = 1;
                    end
                end
            else
                obj.connectivity = defaults.connectivity;
            end
            try
                obj.dataXYZ = obj.streamHandle.dataInXYZ;
            end
            obj.channelIndex = defaults.channels; 
            if isfield(defaults,'color'), obj.color = defaults.color;end
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.showChannelNumber = obj.showChannelNumber;
            defaults.lineColor = obj.lineColor;
            defaults.floorColor = obj.floorColor;
            defaults.background = obj.background;
            defaults.lineWidth = obj.lineWidth;
            defaults.label = obj.label;
            defaults.beep = obj.beep;
            defaults.browserType = 'mocapBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function t0 = createGraphicObjects(obj,nowCursor)
     
            createGraphicObjects@browserHandle(obj);
            box(obj.axesHandle,'off');
            axis(obj.axesHandle,'vis3d')
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor);
            
            if isempty(obj.dataXYZ)
                perm = [1 2 3];
                if (strcmpi(obj.streamHandle.hardwareMetaData.name,'phasespace') || isempty(obj.streamHandle.hardwareMetaData.name)) %&& isa(obj.streamHandle.hardwareMetaData,'hardwareMetaData')
                    perm = [1 3 2];
                elseif isempty(obj.streamHandle.hardwareMetaData.name) || ~isempty(strfind(obj.streamHandle.hardwareMetaData.name,'KinectMocap'))
                    perm = [1 3 2];
                end
                obj.streamHandle.reshape([obj.dim(1) 3 obj.dim(2)/3]);
                data = squeeze(obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t0),perm,obj.channelIndex))';
                obj.streamHandle.reshape(obj.dim);
            else
                data = squeeze(obj.dataXYZ(obj.timeIndex(t0),:,obj.channelIndex))';
            end
            if obj.numberOfChannelsToPlot==1, data = data';end
            if ~strcmp(obj.streamHandle.hardwareMetaData.name,'optitrack')
                indZeros = any(data==0,2);
                data(indZeros,:) = NaN;
            else
                indZeros = false(size(data,1),1);
            end
            
            cla(obj.axesHandle);
            hold(obj.axesHandle,'on')
            axis(obj.axesHandle,'equal')
            obj.markerHandle = zeros(1,obj.numberOfChannelsToPlot);
            [x,y,z] = meshgrid(obj.roomSize.x,obj.roomSize.y,obj.roomSize.z(1));
            surf(obj.axesHandle,double(x),double(y),double(z),'FaceColor',obj.floorColor);
            for it=1:obj.numberOfChannelsToPlot
                obj.markerHandle(it) = scatter3(obj.axesHandle,data(it,1),data(it,2),data(it,3),'filled');                
                set(obj.markerHandle(it),'Tag',['Node ' num2str(obj.channelIndex(it))],'HitTest','off','ButtonDownFcn',...
                    @nodePressed,'MarkerFaceColor',obj.color(it,:),'MarkerEdgeColor','k');
            end
            hold(obj.axesHandle,'off')
            
            ind = find(tril(obj.connectivity));
            if ~isempty(ind)
                [i_ind,j_ind] = ind2sub(size(obj.connectivity),ind);
                I = ismember(i_ind,obj.channelIndex) & ismember(j_ind,obj.channelIndex);
                i_ind(~I) = [];
                j_ind(~I) = [];
                [~,i_ind] = ismember(i_ind,obj.channelIndex);
                [~,j_ind] = ismember(j_ind,obj.channelIndex);
                obj.lineHandle = zeros(length(i_ind),1);
                for it = 1:length(i_ind)
                    x1 = data(i_ind(it),1);
                    x2 = data(j_ind(it),1);
                    y1 = data(i_ind(it),2);
                    y2 = data(j_ind(it),2);
                    z1 = data(i_ind(it),3);
                    z2 = data(j_ind(it),3);
                    obj.lineHandle(it) = line([x1 x2],[y1 y2],[z1 z2],'Color',obj.lineColor,'LineWidth',obj.lineWidth,'Parent',obj.axesHandle);
                    set(obj.lineHandle(it),'Tag',['line ' num2str(it)],'HitTest','off','ButtonDownFcn',...
                        @linePressed,'userData',[i_ind(it) j_ind(it)]);
                end
            end
            
            if obj.showChannelNumber
                I = ~indZeros;
                obj.textHandle = text(double(data(I,1)*1.01),double(data(I,2)*1.01),double(data(I,3)*1.01),obj.label(I),'Color',[1 0 1],'Parent',obj.axesHandle,'FontSize',obj.font.size,'FontWeight','bold');
            end
            
            if obj.onscreenDisplay && ~isempty(obj.streamHandle.event.latencyInFrame)
                loc = any(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) >= obj.passCursor &...
                    obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) <= obj.nowCursor);
                loc = loc | any(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) >= obj.nowCursor &...
                    obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) <= obj.passCursor);
                if loc
                    try
                        play(obj.beep);
                    catch ME
                        disp([ME.identifier '. ' ME.message '\nThe option ''play sound when event'' will be disabled.']);
                        obj.onscreenDisplay = false;
                    end
                end
            end
            obj.passCursor = obj.nowCursor;
            
            xlabel(obj.axesHandle,'x','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,'y','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            zlabel(obj.axesHandle,'z','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            title(obj.axesHandle,'');
            
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec'],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            
            set(obj.axesHandle,'xlim',obj.roomSize.x,'ylim',obj.roomSize.y,'zlim',obj.roomSize.z,'ZGrid','on','YGrid','on',...
                'view',[38 36],'Color',obj.background,'FontSize',obj.font.size,'FontWeight',obj.font.weight)
%             ,'ALimMode','manual','CameraPositionMode','manual','CameraTargetMode','manual','CameraUpVectorMode','manual','CameraViewAngleMode','manual',...
%                 'CLimMode','manual','DataAspectRatioMode','manual','TickDirMode','manual','XTickLabelMode','manual','XTickMode','manual',...
%                 'YTickLabelMode','manual','YTickMode','manual','ZTickLabelMode','manual','ZTickMode','manual');          
        end
        %%
        function t0 = plotThisTimeStamp(obj,nowCursor)
            
            if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
                nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));
            
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),obj.nowCursor);
            if isempty(obj.dataXYZ)
                perm = [1 2 3];
                if (strcmpi(obj.streamHandle.hardwareMetaData.name,'phasespace') || isempty(obj.streamHandle.hardwareMetaData.name)) %&& isa(obj.streamHandle.hardwareMetaData,'hardwareMetaData')
                    perm = [1 3 2];
                elseif isempty(obj.streamHandle.hardwareMetaData.name) || ~isempty(strfind(obj.streamHandle.hardwareMetaData.name,'KinectMocap'))
                    perm = [1 3 2];
                end
                
                obj.streamHandle.reshape([obj.dim(1) 3 obj.dim(2)/3]);
                data = squeeze(obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t0),perm,obj.channelIndex))';
                obj.streamHandle.reshape(obj.dim);
            else
                data = squeeze(obj.dataXYZ(obj.timeIndex(t0),:,obj.channelIndex))';
            end
            if obj.numberOfChannelsToPlot==1, data = data';end
            if ~strcmp(obj.streamHandle.hardwareMetaData.name,'optitrack')
                indZeros = any(data==0,2);
                data(indZeros,:) = NaN;
            else
                indZeros = false(size(data,1),1);
            end
            I2 = find(~indZeros);
% 
%             for it=1:obj.numberOfChannelsToPlot
%                 set(obj.markerHandle(it),'XData',data(it,1),'YData',data(it,2),'ZData',data(it,3))
%             end
            set(obj.markerHandle,{'XData'},num2cell(data(:,1)),{'YData'},num2cell(data(:,2)),{'ZData'},num2cell(data(:,3)))

            ind = find(tril(obj.connectivity));
            if ~isempty(ind)
                [i_ind,j_ind] = ind2sub(size(obj.connectivity),ind);
                I = ismember(i_ind,obj.channelIndex) & ismember(j_ind,obj.channelIndex);
                i_ind(~I) = [];
                j_ind(~I) = [];
                [~,i_ind] = ismember(i_ind,obj.channelIndex);
                [~,j_ind] = ismember(j_ind,obj.channelIndex);
                for it = 1:length(i_ind)
                    x1 = data(i_ind(it),1);
                    x2 = data(j_ind(it),1);
                    y1 = data(i_ind(it),2);
                    y2 = data(j_ind(it),2);
                    z1 = data(i_ind(it),3);
                    z2 = data(j_ind(it),3);
                    set(obj.lineHandle(it),'XData',[x1 x2],'YData',[y1 y2],'ZData',[z1 z2]);
                end
            end
            
            if obj.showChannelNumber
%                 deltax = 0.1*std(data(I2,1));deltax(isnan(deltax)) = eps;
%                 deltay = 0.1*std(data(I2,2));deltay(isnan(deltay)) = eps;
%                 deltaz = 0.1*std(data(I2,3));deltaz(isnan(deltaz)) = eps;
                try delete(obj.textHandle);end %#ok
                obj.textHandle = text(double(data(I2,1)*1.01),double(data(I2,2)*1.01),double(data(I2,3)*1.01),obj.label(I2),'Color',[1 0 1],'Parent',obj.axesHandle,'FontSize',obj.font.size);
            end
            
            if obj.onscreenDisplay && ~isempty(obj.streamHandle.event.latencyInFrame)
                loc = any(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) >= obj.passCursor &...
                    obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) <= obj.nowCursor);
                loc = loc | any(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) >= obj.nowCursor &...
                    obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame) <= obj.passCursor);
                if loc
                    try
                        play(obj.beep);
                    catch ME
                        disp([ME.identifier '. ' ME.message '\nThe option ''play sound when event'' will be disabled.']);
                        obj.onscreenDisplay = false;
                    end
                end
            end
            obj.passCursor = obj.nowCursor;
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
            if any(obj.connectivity(:))
                I = triu(obj.connectivity);
                I = find(I);
                [indI,indJ] = ind2sub(size(obj.connectivity),I);
                if isa(obj.streamHandle,'mocap') && (isempty(obj.streamHandle.animationParameters.conn) || ~isempty(setdiff([indI,indJ],obj.streamHandle.animationParameters.conn,'rows')))
                    obj.streamHandle.animationParameters.conn = [indI,indJ];
                end
            end
            try delete(obj.figureHandle);end %#ok
            if ~isa(obj.master,'browserHandleList')
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
                obj.color(ind,:) = ones(length(index),1)*newColor;
            end
        end
        %%
        function obj = changeSettings(obj)
            h = mocapBrowserSettings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end %#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.speed = userData.speed;
            obj.showChannelNumber = userData.showChannelNumber;
            obj.channelIndex = userData.channels;
            obj.onscreenDisplay = userData.onscreenDisplay;
            if length(userData.floorColor) == 3 && isnumeric(userData.floorColor) 
                obj.floorColor = userData.floorColor;
            end
            if ~isempty(userData.newColor)
                obj.changeChannelColor(userData.newColor.color,userData.newColor.channel)
            end
            if ~isempty(userData.lineWidth) && isnumeric(userData.lineWidth)
                obj.lineWidth = userData.lineWidth;
            end
            if ~isempty(userData.lineColor)
                obj.lineColor = userData.lineColor;
            end
            if ~isempty(userData.background)
                obj.background = userData.background;
            end
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
        %%
        function updateChannelDependencies(~,evnt)
            evnt.AffectedObject.numberOfChannelsToPlot = length(evnt.AffectedObject.channelIndex);
            if evnt.AffectedObject.numberOfChannelsToPlot ~= size(evnt.AffectedObject.color,1)
                evnt.AffectedObject.color = ones(evnt.AffectedObject.numberOfChannelsToPlot,1)*[0.9412 0.9412 0.9412];
            end
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
                evnt.AffectedObject.passCursor = -1;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
        %%
        function updateColorDependencies(~,evnt)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
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

%% ----- marker and line Callbacks -----
function nodePressed(hObject,~)
fig = gcbf;
obj = get(fig,'UserData');
if ~isempty(obj.firstNode),
    try
        A = get(obj.firstNodeObject); %#ok
    catch %#ok
        obj.firstNode = [];
        obj.firstNodeObject = [];
    end
end
firstNode = obj.firstNode;
firstNodeObject = obj.firstNodeObject;
tag = get(gco,'Tag');
currentNode = str2double(tag(6:end));
set(hObject,'MarkerFaceColor','r');

if isempty(firstNode)
    firstNode = currentNode;
    firstNodeObject = hObject;
else
    if currentNode == firstNode, return;end
    obj.connectivity(currentNode,firstNode) = 1;
    obj.connectivity(firstNode,currentNode) = 1;
    %obj.streamHandle.animationParameters.conn = unique([obj.streamHandle.animationParameters.conn;[firstNode,currentNode]],'rows');
    
    x1 = get(obj.firstNodeObject,'XData');
    y1 = get(obj.firstNodeObject,'YData');
    z1 = get(obj.firstNodeObject,'ZData');
    x2 = get(hObject,'XData');
    y2 = get(hObject,'YData');
    z2 = get(hObject,'ZData');
    if ~isempty(obj.lineHandle')
        obj.lineHandle(end+1) = line([x1 x2],[y1 y2],[z1 z2],'Color',obj.lineColor);
    else
        obj.lineHandle = line([x1 x2],[y1 y2],[z1 z2],'Color',obj.lineColor);
    end
    set(obj.lineHandle(end),'Tag',['line ' num2str([firstNode currentNode])],'HitTest','off',...
        'ButtonDownFcn',@linePressed,'UserData',[firstNode currentNode],'LineWidth',obj.lineWidth);
    set(hObject,'MarkerFaceColor',obj.color(currentNode,:));
    set(firstNodeObject,'MarkerFaceColor',obj.color(firstNode,:));
    firstNode = [];
end
obj.firstNode = firstNode;
obj.firstNodeObject = firstNodeObject;
end


function linePressed(hObject,~)
fig = gcbf;
obj = get(fig,'UserData');
ind = get(hObject,'UserData');
obj.connectivity(ind(1),ind(2)) = 0;
obj.connectivity(ind(2),ind(1)) = 0;
delete(hObject);
obj.lineHandle(obj.lineHandle==hObject) = [];
end