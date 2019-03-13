classdef CoreBrowser < handle
    properties
        step = 5;
        eventColor  % onscreen display color
        showEvents = true;
        nowCursor
        mode = -1;
        uuid
        %dcmHandle
        gObjHandle
        axesHandle
        figureHandle
        sliderHandle
        streamHandle
        cursorHandle
        timeTexttHandle
        timerObj = [];
        timeStamp
        font = struct('size',7,'weight','normal');
        eventObj
        master
    end
    properties(SetObservable)
        state = false
        timeIndex
        channelIndex
    end
    properties(Dependent)
        speed;
    end
    properties(SetAccess=private)
        timerPeriod = 0.1;
    end
    properties(Hidden=true)
        %timerMult = 0.008;
        timerMult = 0.01;
        eventLatencyLookUp
    end
    %%
    methods
        function obj = CoreBrowser
            obj = obj@handle;
            obj.speed = 1;
            obj.timerObj = timer('TimerFcn',{@playCallback, obj}, 'Period', obj.timerPeriod,...
                'BusyMode','queue','ExecutionMode','fixedRate');
            obj.addlistener('timeIndex','PostSet',@CoreBrowser.setTimeStamps);
        end
        %%
        function set.speed(obj, val)
            obj.timerPeriod = interp1(1:5,1./linspace(1,10,5),val,'nearest','extrap');
            if strcmp(get(obj.timerObj,'Running'),'on')
                stop(obj.timerObj);
                set(obj.timerObj,'Period', obj.timerPeriod);
                start(obj.timerObj);
            else
                set(obj.timerObj,'Period', obj.timerPeriod);
            end
        end
        function val = get.speed(obj)
            val = interp1(1./linspace(1,10,5),1:5,obj.timerPeriod,'nearest','extrap');
        end
        %%
        function delete(obj)
            if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            delete(obj.timerObj);
            if obj.master ~= -1
                obj.master.updateList();
            end
        end
        %%
        function defaults = saveobj(obj)
            defaults.step = obj.step;
            defaults.showEvents = obj.showEvents;
            defaults.nowCursor = obj.nowCursor;
            defaults.speed = obj.speed;
            defaults.font = obj.font;
            defaults.uuid = obj.uuid;
            defaults.mode = obj.mode;
        end
        %%
        function init(obj)
            if ~isempty(obj.figureHandle) && isvalid(obj.figureHandle)
                cla(obj.axesHandle)
                return;
            end
            
            obj.timeIndex = 1:length(obj.streamHandle.timeStamp);
            obj.eventLatencyLookUp = griddedInterpolant(obj.timeIndex, obj.streamHandle.timeStamp);
            obj.nowCursor = obj.streamHandle.timeStamp(1)+obj.windowWidth/2;
            obj.channelIndex = 1:obj.dim(1);
            obj.eventObj = obj.streamHandle.event;
            
            resFolder = [fileparts(which('CoreBrowser.m')) filesep 'resources'];
            backgroundColor = [0.93 0.96 1];
            fontColor = [0 0 0.4];
            
            hFigure = figure('Color',backgroundColor);
            hFigure.Position(3:4) = [950 530];
            hAxes = axes(hFigure,'Position',[0.1300 0.2302 0.8184 0.6948], 'Box','on');
            
            imgRev   = imread([resFolder filesep '32px-Gnome-media-seek-backward.svg.png']);
            imgPlay  = imread([resFolder filesep '32px-Gnome-media-playback-start.svg.png']);
            imgPause = imread([resFolder filesep '32px-Gnome-media-playback-pause.svg.png']);
            imgNext  = imread([resFolder filesep '32px-Gnome-media-seek-forward.svg.png']);
            imgPref  = imread([resFolder filesep '32px-Gnome-preferences-system.svg.png']);
            hRev     = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[159      53 40 40],'Callback',@play_rev_Callback,'CData',imgRev);
            hPlay    = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[159+41   53 40 40],'Callback',@play_Callback,    'CData',imgPlay, 'UserData',{imgPlay, imgPause});
            hNext    = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[159+41*2 53 40 40],'Callback',@play_fwd_Callback,'CData',imgNext);
            hPref    = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[159+41*3 53 40 40],'Callback',@settings_Callback,'CData',imgPref);
            hSlider  = uicontrol('Parent', hFigure, 'Style', 'slider','Position',[125.13 31 778.87 16],'Callback',@slider_Callback);
            hText    = uicontrol('Parent', hFigure, 'Style', 'text','Position',[374.13 14 266.87 15],'String','Current latency = ');
            hTextMin = uicontrol('Parent', hFigure, 'Style', 'text','Position',[125 14 100 15],'String',obj.timeStamp(1),'HorizontalAlignment','left');
            hTextMax = uicontrol('Parent', hFigure, 'Style', 'text','Position',[125+680 14 100 15],'String',obj.timeStamp(end),'HorizontalAlignment','right');
            
            hText2   = uicontrol('Parent', hFigure, 'Style', 'text','Position',[438 81 100 13],'String','Go to event');
            hNextEvnt = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[543+41 53 40 40],'Callback',@next_Callback,'CData',imgNext,'TooltipString','Go to next event');
            hRevEvnt = uicontrol('Parent', hFigure, 'Style', 'pushbutton','Position',[543 53 40  40],'Callback',@previous_Callback,'CData',imgRev,'TooltipString','Go to previous event');
            uniqueEvents = obj.eventObj.uniqueLabel;
            uniqueEvents(cellfun(@isempty,uniqueEvents)) = [];
            if ~isempty(uniqueEvents)
                hPopUp = uicontrol('Parent', hFigure, 'Style', 'popup', 'Position',[438 37 100 40],'String',uniqueEvents);
            else
                hPopUp = uicontrol(hFigure,'Position',[438 37 100 40]);
                set([hNextEvnt hRevEvnt hText2 hPopUp],'Visible','off','Enable','off');
            end       
            set([hText hText2 hTextMin hTextMax],'BackgroundColor',backgroundColor);
            set([hRev, hPlay hNext hPref hSlider hText, hRevEvnt, hText2, hPopUp, hNextEvnt, hTextMin hTextMax],'Units','Normalized')
            
            obj.figureHandle = hFigure;
            obj.axesHandle = hAxes;
            obj.timeTexttHandle = hText;
            obj.sliderHandle = hSlider;
            
            hListener = addlistener(hSlider,'ContinuousValueChange',@slider_Callback);
            setappdata(obj.sliderHandle,'sliderListeners',hListener);
            set(hFigure,'userData',obj);

            tbHandle = findall(obj.figureHandle,'Type','uitoolbar');
            saveHandle = findall(tbHandle,'Tag','Standard.SaveFigure');
            set(saveHandle,'ClickedCallback','filemenufcn(gcbf,''FileExportSetup'')')            
            
            set(obj.axesHandle,'FontSize',obj.font.size);
            set(obj.timeTexttHandle,'FontSize',obj.font.size,'ForegroundColor',fontColor);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.figureHandle,'name',['MoBILAB ' class(obj) ': ' obj.streamHandle.name]);
            set(obj.sliderHandle,'Value',obj.nowCursor, 'Min',obj.timeStamp(1),'Max',obj.timeStamp(end));
            set(findobj(obj.figureHandle,'tag','connectLine'),'Visible','off');
            set(findobj(obj.figureHandle,'tag','deleteLine'),'Visible','off');
            if isa(obj.streamHandle,'segmentedStreamInContinuousTime') && ~isa(obj,'cometBrowserHandle2')
                tmpObj = obj.streamHandle.originalStreamObj;
            else
                tmpObj = obj.streamHandle;
            end
            set(obj.sliderHandle,'Max',tmpObj.timeStamp(obj.timeIndex(end)));
            set(obj.sliderHandle,'Min',tmpObj.timeStamp(obj.timeIndex(1)));
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.sliderHandle,'SliderStep',0.001*ones(1,2));
            
            if strcmp(obj.mode,'slave')
                set([hRev, hPlay hNext hPref hSlider hText, hTextMin hTextMax ],'Enable','off');
            end
            
            set(obj.figureHandle,'WindowScrollWheelFcn',@(src, event)onMouseWheelMove(obj,[], event));
            if isa(obj,'mocapBrowserHandle')
                set(findobj(obj.figureHandle,'tag','connectLine'),'Visible','on');
                set(findobj(obj.figureHandle,'tag','deleteLine'),'Visible','on');
            end
        end
        %%
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);
            if ~any(I), return;end
            obj.channelIndex = channelIndex;
        end
        %%
        function initEventColor(obj)
            n = length(obj.eventObj.uniqueLabel);
            if n < 100 
                tmpColor = lines(n);
                obj.eventColor = zeros(length(obj.eventObj.latencyInFrame),3);
                for it=1:n
                    loc = ismember(obj.eventObj.label,obj.eventObj.uniqueLabel(it));
                    obj.eventColor(loc,1) = tmpColor(it,1);
                    obj.eventColor(loc,2) = tmpColor(it,2);
                    obj.eventColor(loc,3) = tmpColor(it,3);
                end
            else
                obj.eventColor = lines(length(obj.eventObj.latencyInFrame));
            end
        end
        %%
        function onMouseWheelMove(obj,~,eventObj)
            % step = -10*obj.speed*(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/obj.streamHandle.samplingRate;%#ok
            step = -(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/obj.streamHandle.samplingRate/2;%#ok
            plotStep(obj,step);%#ok
        end
    end
    %%
    methods(Static)
        function setTimeStamps(~,evnt)
            if any(evnt.AffectedObject.timeIndex==-1), return;end
            evnt.AffectedObject.timeStamp = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex);
        end
    end
    %%
    methods(Abstract)
        plotThisTimeStamp(obj,nowCursor)
        plotStep(obj,step)
        changeSettings(obj)
    end
end
function playCallback(obj, event, bObj) %#ok
    plotThisTimeStamp(bObj,bObj.nowCursor+1);
    drawnow;
end
%%
function play_rev_Callback(hObject, eventdata, handles)
try
    browserObj = get(get(hObject,'parent'),'userData');
    browserObj.plotStep(-browserObj.step);
catch ME
    ME.rethrow;
end
end


% --- Executes on slider movement.
function slider_Callback(hObject, eventdata, handles)
browserObj = get(get(hObject,'parent'),'userData');
newNowCursor = get(hObject,'Value');
browserObj.plotThisTimeStamp(newNowCursor);
end


function play_Callback(hObject, eventdata, handles)
try
    browserObj = get(get(hObject,'parent'),'userData');
    browserObj.state = ~browserObj.state;
    CData = get(hObject,'UserData');
    if browserObj.state
        % Play
        set(hObject,'CData',CData{2});
        start(browserObj.timerObj);
    else
        % Pause
        set(hObject,'CData',CData{1});
        stop(browserObj.timerObj);
    end
catch ME
    ME.rethrow;
end
end


function play_fwd_Callback(hObject, eventdata, handles)
try
    browserObj = get(get(hObject,'parent'),'userData');
    browserObj.plotStep(browserObj.step);
catch ME
    ME.rethrow;
end
end


function settings_Callback(hObject, eventdata, handles)
try
    browserObj = get(get(hObject,'parent'),'userData');
    browserObj.changeSettings;
catch ME
    ME.rethrow;
end
end




% --- Executes on button press in previous.
function previous_Callback(hObject, eventdata)
browserObj = get(get(hObject,'parent'),'userData');
if isa(browserObj,'segmentedStreamBrowserHandle')
    eventObj = browserObj.eventObj;
else
    eventObj = browserObj.streamHandle.event;
end
ind = get(findobj(get(hObject,'parent'),'style','popupmenu'),'Value');
uniqueEvents = eventObj.uniqueLabel;
uniqueEvents(cellfun(@isempty,uniqueEvents)) = [];
if ~isempty(uniqueEvents)
    [~,loc] = ismember( eventObj.label, uniqueEvents{ind});
    tmp  = eventObj.latencyInFrame(logical(loc));
    tmp2 = browserObj.eventLatencyLookUp(eventObj.latencyInFrame(logical(loc))) -  browserObj.nowCursor;
    tmp(tmp2>0) = [];
    tmp2(tmp2>=0) = [];
    [~,loc1] = max(tmp2);
    jumpLatency = tmp(loc1);
    if ~isempty(jumpLatency)
        if browserObj.master == -1
            set(browserObj.sliderHandle,'Value',browserObj.eventLatencyLookUp(jumpLatency));
            slider_Callback(browserObj.sliderHandle,eventdata);
        else
            browserObj.master.plotThisTimeStamp(browserObj.eventLatencyLookUp(jumpLatency));
        end
    end
end
end



% --- Executes on button press in next.
function next_Callback(hObject, eventdata)
browserObj = get(get(hObject,'parent'),'userData');
if isa(browserObj,'segmentedStreamBrowserHandle')
    eventObj = browserObj.eventObj;
else
    eventObj = browserObj.streamHandle.event;
end
ind = get(findobj(get(hObject,'parent'),'style','popupmenu'),'Value');
uniqueEvents = eventObj.uniqueLabel;
uniqueEvents(cellfun(@isempty,uniqueEvents)) = [];
if ~isempty(uniqueEvents)
    [~,loc] = ismember( eventObj.label, uniqueEvents{ind});
    tmp  = eventObj.latencyInFrame(logical(loc));
    tmp2 = browserObj.eventLatencyLookUp(eventObj.latencyInFrame(logical(loc))) -  browserObj.nowCursor;
    tmp(tmp2<=0) = [];
    tmp2(tmp2<=0) = [];
    [~,loc1] = min(tmp2);
    jumpLatency = tmp(loc1);
    if ~isempty(jumpLatency)
        if browserObj.master == -1
            set(browserObj.sliderHandle,'Value',browserObj.eventLatencyLookUp(jumpLatency));
            slider_Callback(browserObj.sliderHandle,eventdata);
        else
            browserObj.master.plotThisTimeStamp(browserObj.eventLatencyLookUp(jumpLatency));
        end
    end
end
end
