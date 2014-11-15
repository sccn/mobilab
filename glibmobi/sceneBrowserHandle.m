classdef sceneBrowserHandle < browserHandle
    properties
        gObjHandle
        windowWidth
        axesSize
        channelIndex
        kernelSize = 9;
        kernel
        cursor
        cursorSize
    end
    methods
        %% constructor
        function obj = sceneBrowserHandle(dStreamObj,defaults)
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
            defaults.channels = 9:10;
            defaults.gui = @streamBrowserNG;
            
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
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.channelIndex = defaults.channels; 
            obj.axesSize = ceil(prctile(dStreamObj.data(:,defaults.channels),95));
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = defaults.gui(obj);
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            
            %obj.kernel = fspecial('gaussian',obj.kernelSize,obj.kernelSize);
            obj.kernel = fspecial('disk',obj.kernelSize);
            %obj.kernel = fspecial('log',obj.kernelSize,100);
                                    
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor+obj.windowWidth/2));
            data = ceil(obj.streamHandle.data(obj.timeIndex(t1):obj.timeIndex(t2)-1,obj.channelIndex));
            I = data(:,1) < 1 | data(:,1) > obj.axesSize(1) | data(:,2) < 1 | data(:,2) > obj.axesSize(2);
            data(I,:) = [];
            npoints = size(data,1);
            data = [data (1:npoints)'];
            ind = sub2ind([obj.axesSize npoints],data(:,1),data(:,2),data(:,3));
            nanind = isnan(ind);
            ind(nanind) = [];
            win = gausswin(size(data,1));
            win(nanind) = [];
            screen = zeros([obj.axesSize npoints]);
            screen(ind) = 100*win;
            map = mean(screen,3);
            %map = integralFilter(map, obj.kernel);
            map = imfilter(map, obj.kernel);
            map(map==0) = min(map(:));
            cla(obj.axesHandle);
            obj.gObjHandle = imagesc(map','Parent',obj.axesHandle);
            hold(obj.axesHandle,'on')
            t0 = fix(npoints/2);
            obj.cursor(1) = plot(data(t0,1),data(t0,2),'c+','MarkerSize',obj.cursorSize,'LineWidth',2);
            obj.cursor(2) = plot(data(t0,1),data(t0,2),'co','MarkerSize',obj.cursorSize,'LineWidth',2);
            hold(obj.axesHandle,'off')
            xlabel(obj.axesHandle,'x','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,'z','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            title(obj.axesHandle,'')
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor) ' sec']);
            set(obj.axesHandle,'YDir','normal','xlim',[1 obj.axesSize(1)-obj.kernelSize],'ylim',[1 obj.axesSize(2)-obj.kernelSize],'FontSize',obj.font.size,'FontWeight',obj.font.weight,'CLim',[0 npoints]);
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
            
            data = ceil(obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1):obj.timeIndex(t2)-1,obj.channelIndex));
            I = data(:,1) < 1 | data(:,1) > obj.axesSize(1) | data(:,2) < 1 | data(:,2) > obj.axesSize(2);
            data(I,:) = [];
            
            npoints = size(data,1);
            data = [data (1:npoints)'];
            ind = sub2ind([obj.axesSize npoints],data(:,1),data(:,2),data(:,3));
            nanind = isnan(ind);
            ind(nanind) = [];
            win = gausswin(size(data,1),5);
            win(nanind) = [];
            %win(1:ceil(end/4)) = 0;
            %win(end-ceil(end/4):end) = 0;
            
            screen = zeros([obj.axesSize npoints]);
            screen(ind) = 10000*win;
            map = mean(screen,3);
            %map = integralFilter(map, obj.kernel);
            map = imfilter(map, obj.kernel);
            map(map==0) = min(map(:));
            map = map - min(map(:));
            map = map/max(map(:));
            map = map*npoints;
            set(obj.gObjHandle,'CData',map');
            t0 = fix(npoints/2);
            set(obj.cursor,'XData',data(t0,1),'YData',data(t0,2));
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.axesHandle,'xlim',[1 size(map,1)],'ylim',[1 size(map,2)]);
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