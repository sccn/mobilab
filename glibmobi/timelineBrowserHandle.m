classdef timelineBrowserHandle < browserHandle
    properties
        gObjHandle
        windowWidth
        axesSize
        channelIndex
        kernelSize = 9;
        kernel
        cursor
        cursorSize
        numPlot
        latencies
        colors
        partOfMultiStream
        spaceLocations
        ylim2
    end
    methods
        %% constructor
        function obj = timelineBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.showChannelNumber = false;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + 2.5;
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + 2.5;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'partOfMultiStream'), defaults.partOfMultiStream = 0;end
            
            
            if defaults.partOfMultiStream
                defaults.windowWidth = defaults.endTime-defaults.startTime;
            end
            
            defaults.channels = 1:dStreamObj.numberOfChannels;
            defaults.gui = @streamBrowserNG;
            
            obj.partOfMultiStream = defaults.partOfMultiStream;
            obj.uuid = defaults.uuid;
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            %obj.addlistener('channelIndex','PostSet',@streamBrowserHandle.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@streamBrowserHandle.updateTimeIndexDenpendencies);
            obj.addlistener('font','PostSet',@streamBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
            obj.cursorSize = 50;
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            %obj.timeStamp = obj.streamHandle.timeStamp(obj.timeIndex);
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.channelIndex = defaults.channels;
            %numberOfScreens = length(defaults.channels)/2;
            %obj.axesSize = ceil(prctile(dStreamObj.data(:,defaults.channels),95));
            %obj.axesSize = [numberOfScreens*150 256]; %600 (horizontal) 1024 (vertical) are individual screen sizes
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end;
            
            
            
            
            
            
            [FileName,PathName] = uigetfile2('','');%({'*.txt;*.text','Text File'},'Select the text file');
            if ~any([isnumeric(FileName) isnumeric(PathName)]) 
                [queries, obj.spaceLocations] = extractQueryStringsFromTextFile(PathName,FileName);
                if isempty(queries)
                    disp('Text file cannot start with a blank line'); return;
                end
                obj.numPlot = length(queries);
                obj.latencies = cell(obj.numPlot,1);
                
                hedtags = obj.streamHandle.event.hedTag;
                hedManagerObj = hedManager;
                disp('Finding event latencies...')
                for i = 1:length(obj.latencies)
                    answerArray = hedManagerObj.stringArrayMatchesQueryString(hedtags(:), queries{i}{1});
                    obj.latencies{i}.startLatencies = obj.streamHandle.timeStamp(answerArray);
                    answerArray = hedManagerObj.stringArrayMatchesQueryString(hedtags(:), queries{i}{2});
                    obj.latencies{i}.endLatencies = [obj.streamHandle.timeStamp(answerArray) obj.streamHandle.timeStamp(end)];
                end
            
            else
                disp('You must provide a text file.');return;
            
            end
             
            
            
            obj.colors = ['b','g','r', 'c','m'];
            obj.figureHandle = defaults.gui(obj);
            
            
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.axesHandle,'drawmode','fast');
            
            %obj.kernel = fspecial('gaussian',obj.kernelSize,obj.kernelSize);
            obj.kernel = fspecial('disk',obj.kernelSize);
            %obj.kernel = fspecial('log',obj.kernelSize,100);
                                    
            % find now cursor index
            obj.nowCursor = nowCursor;
            
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            %t1 = binary_findClosest(obj.timeStamp,(obj.nowCursor-obj.windowWidth/2));
            %t2 = binary_findClosest(obj.timeStamp,(obj.nowCursor+obj.windowWidth/2));
            %data = obj.streamHandle.data(obj.timeIndex(t1):obj.timeIndex(t2)-1,obj.channelIndex);
            diff = cell(obj.numPlot,1);
            for i = 1:obj.numPlot
                for j = 1:length(obj.latencies{i}.startLatencies)
                    tmp = obj.latencies{i}.endLatencies-obj.latencies{i}.startLatencies(j);
                    diff{i} = [diff{i} min(tmp(tmp>0))];
                end
            end
            
            %cla(obj.axesHandle)
            hold(obj.axesHandle,'on')
            yValues = (1:obj.numPlot)-0.5;
            for i = 1:length(obj.spaceLocations)
                tmp = obj.spaceLocations(i);
                yValues(tmp+1:end) = 1 + yValues(tmp+1:end); 
            end
            
            for i = 1:obj.numPlot
                for j = 1:length(obj.latencies{i}.startLatencies)
                    %obj.gObjHandle = rectangle('Position',[(obj.latencies{i}.startLatencies(j)),yValues(i),(diff{i}(j)),1],'FaceColor',obj.colors(rem(i,length(obj.colors))+1),'EdgeColor','None');
                    obj.gObjHandle{i}{j} = patch([(obj.latencies{i}.startLatencies(j)),(obj.latencies{i}.startLatencies(j)),...
                        (obj.latencies{i}.startLatencies(j))+(diff{i}(j)),(obj.latencies{i}.startLatencies(j))+(diff{i}(j))],...
                        [yValues(i),yValues(i)+1,yValues(i)+1,yValues(i)],obj.colors(rem(i,length(obj.colors))+1),'EdgeColor','None','FaceAlpha',0.5);
                    plot([(obj.latencies{i}.startLatencies(j)) (obj.latencies{i}.startLatencies(j))],[0 yValues(end)+1.5], 'Color', [0,0,0], 'LineWidth', .01)
                end
            end
            if obj.partOfMultiStream
                set(obj.axesHandle,'xlim',[0 obj.windowWidth]);
                for i = 1:obj.numPlot
                    for j = 1:length(obj.latencies{i}.startLatencies)
                        set(obj.gObjHandle{i}{j},'ButtonDownFcn',@(src,event) patchButtonDownFcn(src,event,obj))
                    end
                end
            else
                set(obj.axesHandle,'xlim',[obj.nowCursor-obj.windowWidth/2 obj.nowCursor+obj.windowWidth/2]);
            end
            %set(obj.axesHandle,'ylim',[0 obj.numPlot+1]);
            set(obj.axesHandle,'ylim',[0 yValues(end)+1.5]);
            obj.ylim2 = yValues(end)+1.5;
            set(obj.axesHandle,'View',[0 90])
            %set(obj.axesHandle,'Ytick',1:obj.numPlot);
            set(obj.axesHandle,'Ytick',yValues+0.5);
            tickstr = cell(obj.numPlot,1);
            for i = 1:obj.numPlot
                tickstr{i} = ['Task ' num2str(i)];
            end
            set(obj.axesHandle,'Yticklabel',tickstr);
            hold(obj.axesHandle,'off')
            
            
%              try delete(obj.cursorHandle.gh);end %#ok
%             obj.cursorHandle.ghIndex = floor(obj.numPlot/2+1);
%             tg = obj.gObjHandle(obj.cursorHandle.ghIndex);
%             obj.cursorHandle.gh = graphics.cursorbar(tg,'Parent',obj.axesHandle);
%             obj.cursorHandle.gh.CursorLineColor = 'r';%[.9,.3,.6]; % default=[0,0,0]='k'
%             obj.cursorHandle.gh.CursorLineStyle = '-.';       % default='-'
%             obj.cursorHandle.gh.CursorLineWidth = 2.5;        % default=1
%             obj.cursorHandle.gh.Orientation = 'vertical';     % =default
%             obj.cursorHandle.gh.TargetMarkerSize = 12;        % default=8
%             obj.cursorHandle.gh.TargetMarkerStyle = 'none';      % default='s' (square)
%             set(obj.cursorHandle.gh.BottomHandle,'MarkerSize',8)
%             set(obj.cursorHandle.gh.TopHandle,'MarkerSize',8)
%             obj.cursorHandle.gh.visible = 'off';
%             set(obj.cursorHandle.gh,'UpdateFcn',@updateCursor);
%             obj.cursorHandle.gh.ShowText = 'on';
%             obj.cursorHandle.gh.Tag = 'graphics.cursorbar';
%             set(get(obj.cursorHandle.gh,'DisplayHandle'),'Visible','off')
            
            % obj.plotThisTimeStamp(nowCursor);

        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
           if ~obj.partOfMultiStream   
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
             
                set(obj.axesHandle,'xlim',[obj.nowCursor-obj.windowWidth/2 obj.nowCursor+obj.windowWidth/2]);
                set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
                set(obj.sliderHandle,'Value',obj.nowCursor);
           else
               obj.nowCursor = nowCursor;set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
               set(obj.axesHandle,'xlim',[0 obj.windowWidth])
           end
           
           lim_y = get(obj.axesHandle,'ylim');
           if lim_y(2)>obj.ylim2
              set(obj.axesHandle,'ylim',[0 obj.ylim2])
           end
        end
        %%
        function patchButtonDownFcn(src,event,obj)
            xdata = get(src,'Xdata');
            if obj.partOfMultiStream
                obj.master.plotThisTimeStamp(xdata(1));
            end
        end
        %%
        function plotStep(obj,step)
            if ~obj.partOfMultiStream
            delta = obj.windowWidth/2;
            if obj.nowCursor+step+delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    obj.nowCursor+step-delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step+delta > obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end))-delta;
            else
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1))+delta;
            end
            else
                newNowCursor = obj.nowCursor;
            end
            obj.plotThisTimeStamp(newNowCursor);
        end
        %%
        function obj = changeSettings(obj)
            if ~obj.partOfMultiStream
                sg = sign(double(obj.speed<1));
                sg(sg==0) = -1;
                speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
                speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
                
                prefObj = [...
                    PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                    PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                    PropertyGridField('kernelSize',obj.kernelSize,'DisplayName','Kernel size','Description','Width of the convolution kernel in pixels.')...
                    PropertyGridField('cursorSize',obj.cursorSize,'DisplayName','Cursor size','Description','Size of the cursor in pixels.')...
                    ];
                
                % create figure
                f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
                position = get(f,'position');
                set(f,'position',[position(1:2) 385 424]);
                g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
                uiwait(f); % wait for figure to close
                val = g.GetPropertyValues();
                
                obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
                obj.windowWidth = val.windowWidth;
                obj.kernelSize = val.kernelSize;
                obj.cursorSize = val.cursorSize;
                figure(obj.figureHandle);
                obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
                obj.createGraphicObjects(obj.nowCursor);
                
                if isa(obj.master,'browserHandleList')
                    obj.master.bound = max([obj.master.bound obj.windowWidth]);
                    obj.master.nowCursor = obj.windowWidth/2;
                    for it=1:length(obj.master.list)
                        if obj.master.list{it} ~= obj, obj.master.list{it}.nowCursor = obj.master.nowCursor;end
                    end
                    obj.master.plotThisTimeStamp(obj.master.nowCursor);
                end
            end
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
    end
end