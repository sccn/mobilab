classdef eegplotNGHandle < browserHandle
    properties
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
        roiObj
        makeSegmentHandle
        isEpoched
        dim
        eventObj
        zoomHandle
        axesSize
    end
    properties(SetObservable)
        channelIndex
        color
    end
    methods
        %% constructor
        function obj = eegplotNGHandle(EEG)
            obj.dim = size(EEG.data);
            obj.isEpoched = false;
            if length(obj.dim) > 2
                obj.isEpoched = true;
            end
            if obj.isEpoched
                EEG.data = reshape(EEG.data,[obj.dim(1) prod(obj.dim(2:3))]);
            end
            
            ntimePoints = size(EEG.data,2);
            obj.streamHandle.name = EEG.setname;
            obj.streamHandle.timeStamp = (0:ntimePoints-1)/EEG.srate;
            obj.streamHandle.numberOfChannels = obj.dim(1);
            obj.streamHandle.mmfName = [pwd filesep obj.streamHandle.name '.bin'];
            precision = class(EEG.data(1));
            fid = fopen(obj.streamHandle.mmfName,'w');
            delta = ceil(0.2*obj.streamHandle.numberOfChannels);
            for it=1:delta:obj.streamHandle.numberOfChannels
                if it+delta-1 <=obj.streamHandle.numberOfChannels
                    fwrite(fid,EEG.data(it:it+delta-1,:)',precision);
                else
                    fwrite(fid,EEG.data(it:end,:)',precision);
                end
            end
            fclose(fid);
            obj.streamHandle.mmfObj = memmapfile(obj.streamHandle.mmfName,'Format',{precision [length(obj.streamHandle.timeStamp) obj.streamHandle.numberOfChannels] 'x'},'Writable',false);
            % obj.streamHandle.data = EEG.data(:,1:end)';
            obj.streamHandle.event = event;
            obj.streamHandle.samplingRate = EEG.srate;
            try
                obj.streamHandle.label = {EEG.chanlocs.labels};
            catch %#ok
                for it=1:obj.streamHandle.numberOfChannels, obj.streamHandle.label{it} = num2str(it);end
            end
            while length(obj.streamHandle.label) < obj.streamHandle.numberOfChannels
                obj.streamHandle.label{end+1} = num2str(length(obj.streamHandle.label)+1);
            end
                
            
            obj.streamHandle.chanlocs = EEG.chanlocs;
            
            obj.streamHandle.icawinv = EEG.icawinv;
            obj.streamHandle.icachansind = EEG.icachansind;
            
            obj.streamHandle.container.container.preferences.gui.backgroundColor = [0.93 0.96 1]; % default eeglab's color: [0.66 0.76 1]
            obj.streamHandle.container.container.preferences.gui.buttonColor = [1 1 1];
            obj.streamHandle.container.container.preferences.gui.fontColor = [0 0 0.4];
            obj.streamHandle.container.container.path = fileparts(which('runmobilab'));
            
            if isfield(EEG.event,'type')
                type = {EEG.event.type};
                latency = round(cell2mat({EEG.event.latency}));
                obj.streamHandle.event = obj.streamHandle.event.addEvent(latency,type);
            end
            
            defaults.startTime = obj.streamHandle.timeStamp(1);
            defaults.endTime = obj.streamHandle.timeStamp(end);
            defaults.gain = 0.25;
            defaults.normalizeFlag = false;
            defaults.step = 1;
            defaults.windowWidth = 5;  % 5 seconds;
            defaults.showChannelNumber = false;
            defaults.channels = 1:obj.dim(1);
            defaults.mode = 'standalone';
            defaults.speed = 1;
            defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
            defaults.font = struct('size',12,'weight','normal');
            defaults.gui = @streamBrowserNG;
            defaults.onscreenDisplay = true;
            
            obj.font = defaults.font;
            
            obj.addlistener('channelIndex','PostSet',@eegplotNGHandle.updateChannelDependencies);
            obj.addlistener('timeIndex','PostSet',@eegplotNGHandle.updateTimeIndexDenpendencies);
            obj.addlistener('color','PostSet',@eegplotNGHandle.updateColorInCell);
            obj.addlistener('font','PostSet',@eegplotNGHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.cursorHandle = [];
            obj.state = false;
            obj.normalizeFlag = defaults.normalizeFlag;
            obj.showChannelNumber = false;
            obj.colormap = 'lines';
            obj.textHandle = [];
            
            obj.timeIndex = 1:ntimePoints; 
            obj.gain = defaults.gain;
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.onscreenDisplay = defaults.onscreenDisplay; % onscreen display information (e.g. events, messages, etc)
            
            obj.channelIndex = defaults.channels; 
            obj.eventObj = obj.streamHandle.event;
            obj.initOsdColor;
            
            
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = defaults.gui(obj);
        end
        %%
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
            defaults.normalizeFlag = obj.normalizeFlag;
            defaults.showChannelNumber = obj.showChannelNumber;
            defaults.gain = obj.gain;
            defaults.colormap = obj.colormap;
            defaults.yTickLabel = obj.yTickLabel;
            defaults.osdColor = obj.osdColor;
            defaults.streamName = obj.streamHandle.name;
            defaults.browserType = 'streamBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.figureHandle,'RendererMode','manual')
            set(obj.figureHandle,'Renderer','OpenGL')
            set(obj.axesHandle,'drawmode','fast');
            set(obj.figureHandle,'name','Scroll channel activities -- eegplotNG(EEG)');
            %obj.dcmHandle.Enable = 'off';
            %obj.dcmHandle.removeAllDataCursors;
            %set(obj.cursorHandle,'Visible','off');
            view(obj.axesHandle,[0 90]);
            
            obj.zoomHandle = zoom(obj.figureHandle);
            try
                hcm = uicontextmenu('Parent',obj.zoomHandle);
            catch
                hcm = uicontextmenu('Parent',obj.zoomHandle.FigureHandle);
            end
            uimenu('Parent',hcm,'Label','Zoom In','Callback','','Tag','item1');
            uimenu('Parent',hcm,'Label','Zoom In','Callback','','Tag','item2');
            set(obj.zoomHandle,'ButtonDownFilter',@(src,event)zoom(obj),'Enable','off','UIContextMenu',hcm);
            
            if isempty(obj.makeSegmentHandle)
                p = fileparts(which('runmobilab'));
                if isempty(p)
                    p = fileparts(which('eeglab'));
                    p = [p filesep 'plugins' filesep 'mobilab'];
                end
                path = fullfile(p,'skin');
                CData = imread([path filesep 'mkSegment.png']);
                userData.icons{1} = CData;
                CData = imread([path filesep 'mkSegment_off.png']);
                userData.icons{2} = CData;
                
                toolbarHandle = findall(obj.figureHandle,'Type','uitoolbar');
                
                obj.makeSegmentHandle = uitoggletool(toolbarHandle,'CData',userData.icons{1},'Separator','on','HandleVisibility','off',...
                    'TooltipString','Reject segment','OnCallback',@(src, event)editRoiOn(obj, [], event),...
                    'OffCallback',@(src, event)editRoiOff(obj, [], event),'userData',userData);
                obj.roiObj = timeSeriesRoi(obj.axesHandle);
            end
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            [~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            [~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1:t2),obj.channelIndex);
            
            cla(obj.axesHandle);
            hold(obj.axesHandle,'on');
            obj.gObjHandle = plot(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),data);
            for it=1:obj.numberOfChannelsToPlot
                set(obj.gObjHandle(it),'color',obj.color(it,:),'userData',{obj.streamHandle, obj.channelIndex(it)});
            end
            %- topoplot on ButtonDown event
            set(obj.gObjHandle,'ButtonDownFcn',@gObject_ButtonDownFcn);
            %-
            %obj.cursorHandle = plot(obj.axesHandle,ones(4,1)*obj.nowCursor,linspace(0,10,4),'LineWidth',2,'Color','r');
            %hold(obj.axesHandle,'off');
            
            
            try delete(obj.cursorHandle.gh);end %#ok
            obj.cursorHandle.ghIndex = floor(obj.numberOfChannelsToPlot/2+1);
            tg = obj.gObjHandle(obj.cursorHandle.ghIndex);
            obj.cursorHandle.gh = [];
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
            
            obj.plotThisTimeStamp(nowCursor);
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
            %if strcmp(obj.dcmHandle.Enable,'on')
            %    set(obj.cursorHandle,'XData',ones(4,1)*obj.nowCursor,'YData',linspace(lim(1),lim(2),4),'Visible','on');
            %else
            %    set(obj.cursorHandle,'Visible','off');
            %end
            %obj.dcmHandle.removeAllDataCursors;
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                try delete( findall(obj.axesHandle,'Tag','graphics.cursorbar'));end %#ok
            end
            set(obj.gObjHandle,'XData',obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),{'YData'},data,{'Color'},obj.colorInCell);           
            
            xlim(obj.axesHandle,obj.streamHandle.timeStamp(obj.timeIndex([t1 t2])));
            ylim(obj.axesHandle,lim);  % static limit
            obj.axesSize.xlim = obj.streamHandle.timeStamp(obj.timeIndex([t1 t2]));
            obj.axesSize.ylim = lim;
            
            if obj.onscreenDisplay
                if length(obj.eventObj.latencyInFrame) ~= size(obj.osdColor,1), obj.initOsdColor;end
                [~,loc1,loc2] = intersect(obj.streamHandle.timeStamp(obj.timeIndex(t1:t2)),obj.streamHandle.timeStamp(obj.eventObj.latencyInFrame));
                Nloc1 = length(loc1);
                if ~isempty(loc1)
                    hold(obj.axesHandle,'on');
                    set(obj.figureHandle,'CurrentAxes',obj.axesHandle)
                    ind = obj.timeIndex(t1:t2);
                    linesHandler = line(ones(2,1)*obj.streamHandle.timeStamp(ind(loc1)),(ones(length(loc1),1)*lim)','Parent',obj.axesHandle);
                    %kkIndex = ind(loc1) == 1;
                    
                    textPos = [obj.streamHandle.timeStamp(ind(loc1))-0.5*(obj.streamHandle.timeStamp(ind(loc1))-...
                            obj.streamHandle.timeStamp(ind(loc1)-0));ones(1,Nloc1)*lim(2)*1.01];

                    try delete(obj.textHandle);end %#ok
                    set(linesHandler,{'color'},num2cell(obj.osdColor(loc2,:)',[1 3])');
                    obj.textHandle = zeros(length(loc1),1);
                    for it=1:Nloc1
                        obj.textHandle(it) = text('Position',textPos(:,it),'String',obj.eventObj.label(loc2(it)),'Color',obj.osdColor(loc2(it),:),...
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
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                try
                    obj.cursorHandle.gh = graphics.cursorbar(obj.gObjHandle(obj.cursorHandle.ghIndex),'Parent',obj.axesHandle);
                    obj.cursorHandle.gh.CursorLineColor = 'r';%[.9,.3,.6]; % default=[0,0,0]='k'
                    obj.cursorHandle.gh.CursorLineStyle = '-.';       % default='-'
                    obj.cursorHandle.gh.CursorLineWidth = 2.5;        % default=1
                    obj.cursorHandle.gh.Orientation = 'vertical';     % =default
                    obj.cursorHandle.gh.TargetMarkerSize = 12;        % default=8
                    obj.cursorHandle.gh.TargetMarkerStyle = 'none';      % default='s' (square)
                    set(obj.cursorHandle.gh.BottomHandle,'MarkerSize',8)
                    set(obj.cursorHandle.gh.TopHandle,'MarkerSize',8)
                    obj.cursorHandle.gh.visible = 'on';
                    set(obj.cursorHandle.gh,'UpdateFcn',@updateCursor);
                    obj.cursorHandle.gh.ShowText = 'on';
                    obj.cursorHandle.gh.Tag = 'graphics.cursorbar';
                    set(get(obj.cursorHandle.gh,'DisplayHandle'),'Visible','off') 
                catch ME
                    disp(ME.message)
                    disp('cursorbar functionality needs to be ported over MATLAB 2014b.')
                    obj.cursorHandle.gh = [];
                end
            end
            obj.roiObj.paint;
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
            if ~any(ind), errordlg('Index exceeds the number of channels.');end
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
            try close(get(obj.axesHandle,'userData'));end %#ok
            try delete(obj.figureHandle);end %#ok
            if ~isa(obj.master,'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
            delete(obj.streamHandle.mmfName);
            if isempty(obj.roiObj.segmentObj.startLatency) ||  obj.roiObj.segmentObj.endLatency == 0, return;end
            N = length(obj.roiObj.segmentObj.startLatency);
            TMPREJ = zeros(N,obj.streamHandle.numberOfChannels+5);
            TMPREJ = zeros(N,5);
            TMPREJ(:,1:2) = [obj.roiObj.segmentObj.startLatency(:) obj.roiObj.segmentObj.endLatency(:)]*obj.streamHandle.samplingRate;
            TMPREJ(:,3:5) = ones(N,1)*obj.roiObj.color; 
            assignin('base','TMPREJ',TMPREJ);
        end
        %%
        function obj = changeSettings(obj)
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
            prefObj = [...
                PropertyGridField('gain',obj.gain,'DisplayName','Channel gain','Description','')...
                PropertyGridField('channels',obj.channelIndex,'DisplayName','Channels to plot','Description','This field accept matlab code returning a subset of channels, for instance use: ''setdiff(1:10,[3 5 7])'' to plot channels from 1 to 10 excepting 3, 5, and 7.')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                PropertyGridField('normalizeFlag',obj.normalizeFlag,'DisplayName','Normalize channels','Description','Divides each channels by its standard deviation within the segment to plot.')...
                PropertyGridField('showChannelNumber',obj.showChannelNumber,'DisplayName','Show channel number or label','Description','')...
                PropertyGridField('onscreenDisplay',obj.onscreenDisplay,'Category','Events','DisplayName','Show events','Description','')...
                PropertyGridField('labels', obj.streamHandle.event.uniqueLabel,'Category','Events','DisplayName','Show only a subset of events','Description','')...
                PropertyGridField('colormap',obj.colormap,'Type',PropertyType('char','row',{'lines','eegplot'}),'DisplayName','Colormap','Description','')...
                ];
            
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

            obj.gain = val.gain;
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.windowWidth = val.windowWidth;
            obj.normalizeFlag = val.normalizeFlag;
            obj.showChannelNumber = val.showChannelNumber;
            obj.channelIndex = val.channels;
            obj.onscreenDisplay = val.onscreenDisplay;
            obj.changeColormap(val.colormap);
            
            figure(obj.figureHandle);
            obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
            obj.createGraphicObjects(obj.nowCursor);
        end
        %%
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);
            if ~any(I), return;end
            obj.channelIndex = channelIndex;
        end
        %%
        function editRoiOn(obj,~,~)
            userData = get(obj.makeSegmentHandle,'userData');
            set(obj.makeSegmentHandle,'CData',userData.icons{2});
            obj.zoomHandle.enable = 'off';
            % obj.dcmHandle.enable = 'off';
            obj.roiObj.isactive = true;
            % editRoi(obj);
            obj.roiObj.paint;
        end
        %%
        function editRoi(obj,~,~)
            if ~obj.roiObj.isactive, return;end
            try %#ok
                rect = getrect(obj.axesHandle,'ButtonDown');
                bsObj = basicSegment([rect(1),rect(1)+rect(3)],'tmp');
                
                obj.roiObj.addSegment(bsObj);
                obj.roiObj.paint;
            end
        end
        %%
        function editRoiOff(obj,~,~)
            userData = get(obj.makeSegmentHandle,'userData');
            set(obj.makeSegmentHandle,'CData',userData.icons{1});
            obj.roiObj.isactive = false;
            obj.roiObj.paint;
        end
        %%
        function res = zoom(obj)
            res = true;
            if strcmp(obj.zoomHandle.Enable,'off'), return;end
            direction = obj.zoomHandle.Direction;
            if strcmp(get(obj.figureHandle,'SelectionType'),'alt')
                if strcmp(obj.zoomHandle.Direction,'out')
                    direction = 'in';
                else
                    direction = 'out';
                end
            end
            xl = get(obj.axesHandle,'XLim');
            yl = get(obj.axesHandle,'YLim');
            dx = diff(xl);
            dy = diff(yl);
            if strcmp(direction,'out')
                nxl = [xl(1)-dx/4 xl(2)+dx/4];
                nyl = [yl(1)-dy/4 yl(2)+dy/4];
                if nxl(1) < obj.axesSize.xlim(1) || nxl(2) > obj.axesSize.xlim(2) ||...
                        nyl(1) < obj.axesSize.ylim(1) || nyl(2) > obj.axesSize.ylim(2), nxl = xl;nyl = yl;end
            else
                nxl = [xl(1)+dx/4 xl(2)-dx/4];
                nyl = [yl(1)+dy/4 yl(2)-dy/4];
            end
            set(obj.axesHandle,'XLim',nxl);
            set(obj.axesHandle,'YLim',nyl);
            set(obj.roiObj.axesHandle,'XLim',nxl);
            set(obj.roiObj.axesHandle,'YLim',nyl);
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
        %%
        function updateFont(~,evnt)
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
    end
end

%% -----------------------------
function gObject_ButtonDownFcn(hObject,~,~)
browserObj = get(get(get(hObject,'parent'),'parent'),'userData');
if strcmp(browserObj.zoomHandle.Enable,'on'), return;end
%if strcmp(browserObj.dcmHandle.enable,'on'), return;end
userData = get(hObject,'userData'); 
if ~iscell(userData), return;end
if length(userData)~=2, return;end

tpHandle = get(get(hObject,'parent'),'userData');
try close(tpHandle); end%#ok;
streamHandle = userData{1};
if isempty(streamHandle.icawinv), return;end
tpHandle = figure;
topoplot(streamHandle.icawinv(:,userData{2}),streamHandle.chanlocs);

position = get(tpHandle,'position');
A = get(0,'PointerLocation');
%A = get(get(get(hObject,'parent'),'parent'),'currentpoint');
set(tpHandle,'position',[A 0.25*position(3:4)]);
%set(tpHandle,'position',[position(1:2) 0.25*position(3:4)]);
set(tpHandle,'toolbar','none');
set(tpHandle,'menubar','none');
set(hObject,'userData',userData); 
set(get(hObject,'parent'),'userData',tpHandle);
end