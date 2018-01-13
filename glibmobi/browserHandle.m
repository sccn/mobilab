classdef browserHandle < handle
    properties
        step = 1;
        onscreenDisplay
        nowCursor
        master = [];
        uuid
        %dcmHandle
        axesHandle
        figureHandle
        sliderHandle
        streamHandle
        cursorHandle
        timeTexttHandle
        timerObj = [];
        timeStamp
    end
    properties(SetObservable)
        state = false
        timeIndex
        font
        speed
    end
    properties(Hidden=true)
        %timerMult = 0.008;
        timerMult = 0.01;
    end
    %%
    methods
        function obj = browserHandle
            obj = obj@handle;
            obj.font.size = 12;
            obj.font.weight = 'normal'; % bold
            obj.speed = 1;
            obj.timerObj = timer('TimerFcn',{@playCallback, obj}, 'Period', obj.timerMult/obj.speed,...
                'BusyMode','queue','ExecutionMode','fixedRate','StopFcn',{@stopCallback, obj});
            obj.addlistener('state','PostSet',@browserHandle.triggerTimer);
            obj.addlistener('speed','PostSet',@browserHandle.setTimerPeriod);
            obj.addlistener('timeIndex','PostSet',@browserHandle.setTimeStamps);
        end
        %%
        function delete(obj)
            if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            delete(obj.timerObj);
        end
        %%
        function defaults = saveobj(obj)
            defaults.step = obj.step;
            defaults.onscreenDisplay = obj.onscreenDisplay;
            defaults.nowCursor = obj.nowCursor;
            defaults.speed = obj.speed;
            defaults.font = obj.font;
            defaults.uuid = obj.uuid;
            if isa(obj.master,'browserHandleList')
                defaults.mode = 'slave';
            else defaults.mode = 'standalone';
            end 
        end
        %%
        function createGraphicObjects(obj)
            disp('Creating the figure...');
            tbHandle = findall(obj.figureHandle,'Type','uitoolbar');
            saveHandle = findall(tbHandle,'Tag','Standard.SaveFigure');
            set(saveHandle,'ClickedCallback','filemenufcn(gcbf,''FileExportSetup'')')
            
            obj.cursorHandle.tb = findall(tbHandle,'Tag','Exploration.DataCursor');
            set(obj.cursorHandle.tb,'ClickedCallback',@(src, event)enableCursor(obj, [], event),'TooltipString','Cursor','State','off');
            
            preferences = get_mobilab_preferences;
            if ~isMatlab2014b()          
                set(obj.axesHandle,'drawmode','fast');
            end
            set(obj.axesHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.timeTexttHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight,'ForegroundColor',preferences.gui.fontColor);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.figureHandle,'name',['MoBILAB ' class(obj) ': ' obj.streamHandle.name]);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(findobj(obj.figureHandle,'tag','connectLine'),'Visible','off');
            set(findobj(obj.figureHandle,'tag','deleteLine'),'Visible','off');
            if isa(obj.streamHandle,'segmentedStreamInContinuousTime') && ~isa(obj,'cometBrowserHandle2')
                tmpObj = obj.streamHandle.originalStreamObj;
            else tmpObj = obj.streamHandle;
            end
            set(findobj(obj.figureHandle,'tag','text4'),'String',num2str(tmpObj.timeStamp(obj.timeIndex(1)),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight,...
                'ForegroundColor',preferences.gui.fontColor);
            set(findobj(obj.figureHandle,'tag','text5'),'String',num2str(tmpObj.timeStamp(obj.timeIndex(end)),4),'FontSize',obj.font.size,'FontWeight',obj.font.weight,...
                'ForegroundColor',preferences.gui.fontColor);
            set(obj.sliderHandle,'Max',tmpObj.timeStamp(obj.timeIndex(end)));
            set(obj.sliderHandle,'Min',tmpObj.timeStamp(obj.timeIndex(1)));
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.sliderHandle,'SliderStep',0.001*ones(1,2));
            
            if isa(obj.master,'browserHandleList') || isempty(obj.master)
                set(obj.sliderHandle,'Visible','off')
                fp = get(obj.figureHandle,'position');
                ap = get(obj.axesHandle,'position');
                position = get(findobj(obj.figureHandle,'Tag','text6'),'position');
                set(findobj(obj.figureHandle,'Tag','text6'),'position',[position(1) 0.15*(ap(2)+fp(2)) position(3:4)])
                set(findobj(obj.figureHandle,'Tag','text10'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','uipanel6'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','previous'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','next'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','play_rev'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','play_fwd'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','play'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','settings'),'Visible','off')
                set(findobj(obj.figureHandle,'Tag','listbox1'),'Visible','off')
                set(findobj(obj.figureHandle,'tag','text4'),'Visible','off');
                set(findobj(obj.figureHandle,'tag','text5'),'Visible','off');
            else
                rotate3d(obj.axesHandle,'off');
                set(obj.figureHandle,'WindowScrollWheelFcn',@(src, event)onMouseWheelMove(obj,[], event),'KeyPressFcn',@(src, event)onKeyPress(obj,[], event));
                if isa(obj,'mocapBrowserHandle')
                    set(findobj(obj.figureHandle,'tag','connectLine'),'Visible','on');
                    set(findobj(obj.figureHandle,'tag','deleteLine'),'Visible','on');
                end
                set(obj.sliderHandle,'Visible','on')
                set(findobj(obj.figureHandle,'Tag','text10'),'Visible','on','ForegroundColor',preferences.gui.fontColor)
                set(findobj(obj.figureHandle,'Tag','uipanel6'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','previous'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','next'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','play_rev'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','play_fwd'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','play'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','settings'),'Visible','on')
                set(findobj(obj.figureHandle,'Tag','listbox1'),'Visible','on')
                set(findobj(obj.figureHandle,'tag','text4'),'Visible','on');
                set(findobj(obj.figureHandle,'tag','text5'),'Visible','on');
            end
            if isMatlab2014b()
                set(findobj(obj.figureHandle,'Tag','uipanel6'),'Visible','off')
            end
        end
        %%
        function enableCursor(obj,~,~)
            if strcmp(get(obj.cursorHandle.tb,'State'),'on')
                obj.plotThisTimeStamp(obj.nowCursor);
            else obj.cursorHandle.gh.visible = 'off';
            end
        end
        function play(obj)
            try
                delete(obj.timerObj);
            end
            fs = obj.streamHandle.samplingRate;
            obj.timerObj = timer('TimerFcn',{@playCallback, obj}, 'Period', 1/(fs*obj.speed),...
                'BusyMode','queue','ExecutionMode','fixedRate','StopFcn',{@stopCallback, obj});
            obj.state = ~obj.state; 
        end
        function onMouseWheelMove(obj,~,eventObj)
            % step = -10*obj.speed*(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/obj.streamHandle.samplingRate;%#ok
            step = -(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/obj.streamHandle.samplingRate/2;%#ok
            plotStep(obj,step);%#ok
        end
        function onKeyPress(obj,~,eventObj)
            switch eventObj.Key
                case 'leftarrow',  plotStep(obj,-obj.step*obj.speed*2);
                case 'rightarrow', plotStep(obj,obj.step*obj.speed*2);
            end
        end
    end
    %%
    methods(Static)
        function triggerTimer(~,evnt)
            if evnt.AffectedObject.state && strcmp(get(evnt.AffectedObject.timerObj,'Running'),'off')
                 start(evnt.AffectedObject.timerObj);
            else stop(evnt.AffectedObject.timerObj);
            end
        end
        %%
        function setTimerPeriod(~,evnt)
            if strcmp(get(evnt.AffectedObject.timerObj,'Running'),'on');
                 stop(evnt.AffectedObject.timerObj);
                 triggerFlag = true;
            else triggerFlag = false;
            end
            set(evnt.AffectedObject.timerObj,'Period',evnt.AffectedObject.timerMult/evnt.AffectedObject.speed);
            if triggerFlag, start(evnt.AffectedObject.timerObj);end
        end
        %%
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
if bObj.state % this if statement added so that "pause" works.
    %plotThisTimeStamp(bObj,bObj.nowCursor+bObj.timerMult*bObj.speed);
    plotThisTimeStamp(bObj,bObj.nowCursor+1/bObj.streamHandle.samplingRate);
    drawnow;
else
    bObj.state = false;
    stop(bObj.timerObj);
end
end
%%
function stopCallback(obj, event, bObj) %#ok
playObj = findobj(bObj.figureHandle,'Tag','play');
CData = get(playObj,'UserData');
if bObj.state
    set(playObj,'CData',CData{1});
end
bObj.state = false;
stop(bObj.timerObj);
end