classdef cometBrowserHandle2 < browserHandle
    properties
        markerHandle
        textHandle
        showChannelNumber
        numberOfChannelsToPlot
        colorInCell
        background
        dim
        label
        tail
        time
        xlim
        ylim
        videoHandle
        rec = false
        axisOff = false
        XTickLabel
        XTick
        YTickLabel
        YTick
    end
    properties(SetObservable)
        channelIndex
        color
    end
    methods
        %% constructor
        function obj = cometBrowserHandle2(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.showChannelNumber = false;
                defaults.channels = 1;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
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
            if ~isfield(defaults,'background'), defaults.background = [1 1 1];end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + 2.5;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            
            %if size(dStreamObj,2) ~=  2, error('cometBrowserHandle2 only works with 2D data (xy-coordinates).');end
            if ~isa(dStreamObj,'pcaMocap'), error('cometBrowserHandle2 only works with 2D data (xy-coordinates).');end
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
            obj.tail = round(dStreamObj.samplingRate*1);
            %obj.color = flipud(gray(obj.tail));
             obj.color = gray(obj.tail);
            obj.background = defaults.background;
            obj.dim = [length(obj.streamHandle.timeStamp) obj.streamHandle.numberOfChannels];
            
            obj.timeIndex = [1 size(dStreamObj,1)];
            obj.time = (0:size(dStreamObj,1)-1)/dStreamObj.samplingRate;
            obj.step = defaults.speed;       % one second
            obj.nowCursor = obj.time(1)+2.1;
                                                           
            obj.channelIndex = defaults.channels; 
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            
            mx = 1.5*max(max(max(abs(squeeze(obj.streamHandle.dataInXY(:,:,obj.channelIndex))))));
            obj.xlim = [-mx mx];
            obj.ylim = [-mx mx];
            
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.showChannelNumber = obj.showChannelNumber;
            defaults.color = obj.color;
            defaults.background = obj.background;
            defaults.label = obj.label;
            defaults.streamName = obj.streamHandle.name;
            defaults.browserType = 'cometBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.figureHandle,'Renderer','zbuffer');
            try
                MoBILAB = evalin('base','mobilab');
            catch %#ok
                error('MoBILAB:noRunning','You have to have MoBILAB running.');
            end
            path = fullfile(MoBILAB.path,'skin');
            CData = imread([path filesep '32px-Gnome-media-record.svg.png']);
            recHandle = findobj(obj.figureHandle,'tag','connectLine');
            set(recHandle,'Visible','on','CData',CData,'Callback',@(src, event)writeMovie(obj, '', event),'ToolTip','Movie maker');
            
            set(obj.sliderHandle,'Max',obj.time(end));
            set(obj.sliderHandle,'Min',obj.time(1));
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(findobj(obj.figureHandle,'tag','text4'),'String',num2str(obj.time(1),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(findobj(obj.figureHandle,'tag','text5'),'String',num2str(obj.time(end),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            
            % find now cursor index
            delta = obj.tail/obj.streamHandle.samplingRate;
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.time - (obj.nowCursor-delta/2)));  
            %[~,t2] = min(abs(obj.time - (obj.nowCursor+delta/2)));  
            t1 = binary_findClosest(obj.time,(obj.nowCursor-delta/2));
            t2 = binary_findClosest(obj.time,(obj.nowCursor+delta/2));
            data = obj.streamHandle.dataInXY(t1:t2-1,:,obj.channelIndex);
            
            %if obj.numberOfChannelsToPlot==1, data = data';end
            data(isnan(data(:))) = NaN;     
                       
            delete(obj.markerHandle(ishandle(obj.markerHandle)));
            obj.markerHandle = [];
            cla(obj.axesHandle);
            set(obj.axesHandle,'nextplot','add')
            
            if obj.showChannelNumber
                I = squeeze(sum(data(end,:,:),2))~=0;
                try delete(obj.textHandle);end %#ok
                obj.textHandle = text(double(1.01*data(end,1,I)),double(1.01*data(end,2,I)),obj.label(I),'Color',[1 0 1],'Parent',obj.axesHandle);
            end
            
            for it=1:obj.numberOfChannelsToPlot
                obj.markerHandle(it) = scatter(data(:,1,it),data(:,2,it),'filled','CData',obj.color,'Parent',obj.axesHandle);
            end
              
            xlabel(obj.axesHandle,'x','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            ylabel(obj.axesHandle,'y','Color',[0 0 0.4],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            title(obj.axesHandle,'')
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor) ' sec']);
            
            set(obj.axesHandle,'Color',obj.background,'FontSize',obj.font.size,'FontWeight',obj.font.weight,'view',[0 90],'Xlim',obj.xlim,'Ylim',obj.ylim);
            set(obj.axesHandle,'view',[0 90],'XLimMode','manual','YLimMode','manual');
            
            obj.XTickLabel = get(obj.axesHandle,'XTickLabel');
            obj.XTick = get(obj.axesHandle,'XTick');
            obj.YTickLabel = get(obj.axesHandle,'YTickLabel');
            obj.YTick = get(obj.axesHandle,'YTick');
            
            if obj.axisOff
                obj.XTickLabel = get(obj.axesHandle,'XTickLabel');
                obj.XTick = get(obj.axesHandle,'XTick');
                obj.YTickLabel = get(obj.axesHandle,'YTickLabel');
                obj.YTick = get(obj.axesHandle,'YTick');
                set(obj.axesHandle,'XTickLabel','','XTick',[],'YTickLabel','','YTick',[])
                xlabel(obj.axesHandle,'');
                ylabel(obj.axesHandle,'');
            else
                set(obj.axesHandle,'XTickLabel',obj.XTickLabel,'XTick',obj.XTick,'YTickLabel',obj.YTickLabel,'YTick',obj.YTick)
                xlabel(obj.axesHandle,'x');
                ylabel(obj.axesHandle,'y');
            end
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
            delta = obj.tail/obj.streamHandle.samplingRate;
            if  nowCursor + delta < obj.time(end) && nowCursor - delta > obj.time(1)
                newNowCursor = nowCursor;
            elseif nowCursor + delta > obj.time(end)
                newNowCursor = obj.time(end) - delta;
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            else
                newNowCursor = obj.time(1) + delta;
            end
            nowCursor = newNowCursor;
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.time - (obj.nowCursor-delta))); 
            t1 = binary_findClosest(obj.time,(obj.nowCursor-delta));
            t2 = t1+obj.tail;
            
            data = obj.streamHandle.dataInXY(t1:t2-1,:,obj.channelIndex);
           
            if obj.showChannelNumber
                I = squeeze(sum(data(end,:,:),2))~=0;
                try delete(obj.textHandle);end %#ok
                obj.textHandle = text(double(1.01*data(end,1,I)),double(1.01*data(end,2,I)),obj.label(I),'Color',[1 0 1],'Parent',obj.axesHandle);
            end
            
            data(isnan(data(:))) = NaN;
            for it=1:obj.numberOfChannelsToPlot
                set(obj.markerHandle(it),'XData',data(:,1,it),'YData',data(:,2,it));
            end
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            
            if obj.rec
                try
                frame = getframe(obj.axesHandle);
                writeVideo(obj.videoHandle,frame);
                catch ME
                    close(obj.videoHandle);
                    writeMovie(obj);
                    disp(ME.message);
                end
            end
        end
        %%
        function plotStep(obj,step)
            if obj.nowCursor+step < obj.time(end) && obj.nowCursor+step > obj.time(1)
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step > obj.time(end)
                newNowCursor = obj.time(end);
            else
                newNowCursor = obj.time(1);
            end
            obj.plotThisTimeStamp(newNowCursor);
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
            h = cometBrowser2Settings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end %#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.speed = userData.speed;
            obj.showChannelNumber = userData.showChannelNumber;
            obj.channelIndex = userData.channels;
            obj.xlim =  userData.xlim;
            obj.ylim =  userData.ylim;
            if ~isempty(userData.tail) && isnumeric(userData.tail)
                obj.tail = round(userData.tail);
                obj.color = flipud(gray(obj.tail));
            end
            if ~isempty(userData.background)
                obj.background = userData.background;
            end
            if any(obj.background < 0.4), obj.color = gray(obj.tail);end
            obj.axisOff = userData.axisOff;
            figure(obj.figureHandle);
            obj.createGraphicObjects(obj.nowCursor);
            
        end
        %%
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);
            if ~any(I), return;end
            obj.channelIndex = channelIndex;
        end
        %%
        function writeMovie(obj,file,~)
            if nargin < 2, file = '';end
            obj.rec = ~obj.rec;
            
            if obj.rec
                
                if isempty(file)
                    [name,path] = uiputfile2('unnamed.avi');
                    if ~ischar(path) || ~ischar(name), 
                        obj.rec = ~obj.rec;
                        return;
                    end
                    file = fullfile(path,name);
                end
                obj.videoHandle = VideoWriter(file, 'Uncompressed AVI');
                obj.speed=obj.speed/30;
                obj.step=obj.speed;
                obj.videoHandle.FrameRate = 30;
                open(obj.videoHandle);

                set(findobj(obj.figureHandle,'Tag','previous'),'Enable','off')
                set(findobj(obj.figureHandle,'Tag','next'),'Enable','off')
                set(findobj(obj.figureHandle,'Tag','play_rev'),'Enable','off')
                set(findobj(obj.figureHandle,'Tag','play_fwd'),'Enable','off')
                set(findobj(obj.figureHandle,'Tag','play'),'Enable','off')
                set(findobj(obj.figureHandle,'Tag','settings'),'Enable','off')
                set(obj.figureHandle,'Units','Points');
                set(obj.axesHandle,'Units','Points');
                obj.play;
            else
                obj.play;
                close(obj.videoHandle);
                obj.speed = 1;
                set(findobj(obj.figureHandle,'Tag','previous'),'Enable','on')
                set(findobj(obj.figureHandle,'Tag','next'),'Enable','on')
                set(findobj(obj.figureHandle,'Tag','play_rev'),'Enable','on')
                set(findobj(obj.figureHandle,'Tag','play_fwd'),'Enable','on')
                set(findobj(obj.figureHandle,'Tag','play'),'Enable','on')
                set(findobj(obj.figureHandle,'Tag','settings'),'Enable','on')
                set(obj.figureHandle,'Units','Normalized');
                set(obj.axesHandle,'Units','Normalized');
            end
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
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'String',num2str(evnt.AffectedObject.time(1),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'String',num2str(evnt.AffectedObject.time(end),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            
            xlabel(evnt.AffectedObject.axesHandle,'x','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            zlabel(evnt.AffectedObject.axesHandle,'y','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            ylabel(evnt.AffectedObject.axesHandle,'z','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
    end
end
