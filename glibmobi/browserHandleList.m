classdef browserHandleList < handle
    properties
        step
        master
        slHandle
        ceHandle
        bound
        timeTexttHandle
        sliderHandle
        minTextHandle
        maxTextHandle
        font
        timerObj = [];
        state
    end
    properties(SetObservable)
        speed
        nowCursor
        list
        startTime
        endTime
        playbackStep;
    end
    properties(Constant,Hidden=true)
        %timerMult = 0.008;
        timerMult = 0.1;
    end
    methods
        function obj = browserHandleList(master)
            if nargin < 1, error('Not enough input arguments in browserHandleList constructor.');end
            obj.list = [];
            obj.startTime = -1;
            obj.endTime = 1e6;
            obj.step = 1;
            obj.master = master;
            obj.speed = 1;
            obj.timerObj = timer('TimerFcn',{@playCallback, obj}, 'Period', obj.timerMult/obj.speed,...
                'BusyMode','queue','ExecutionMode','fixedRate','StopFcn',{@stopCallback, obj});
            % obj.addlistener('state','PostSet',@browserHandleList.triggerTimer);
            obj.addlistener('speed','PostSet',@browserHandleList.setTimerPeriod);
            obj.addlistener('nowCursor','PostSet',@browserHandleList.updateNowCursorDependency);
            obj.addlistener('list','PostSet',@browserHandleList.updateListDependency);
            obj.addlistener('startTime','PostSet',@browserHandleList.updateStartTime);
            obj.addlistener('endTime','PostSet',@browserHandleList.updateEndTime);
            obj.nowCursor = 0;
            obj.state = false;
            obj.speed = 1;
            obj.bound = 0;
            obj.timeTexttHandle = findobj(master,'tag','text8');
            obj.sliderHandle = findobj(master,'tag','slider1');
            obj.minTextHandle = findobj(master,'tag','text6');
            obj.maxTextHandle = findobj(master,'tag','text7');
            obj.font.size = 12;
            obj.font.weight = 'normal'; % bold
            
            set(obj.master,'WindowScrollWheelFcn',@(src, event)onMouseWheelMove(obj,[], event),'KeyPressFcn',@(src, event)onKeyPress(obj,[], event));
            set(obj.sliderHandle,'Max',obj.endTime);
            set(obj.sliderHandle,'Min',obj.startTime);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.minTextHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.maxTextHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec'],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
        end
        %%
        function save(obj,sessionFilename)
            N = length(obj.list);
            for it=1:N
                objStruct.list{it} = obj.list{it}.saveobj;
            end
            objStruct.startTime = obj.startTime;
            objStruct.endTime = obj.endTime;
            objStruct.step = obj.step;
            objStruct.nowCursor = obj.nowCursor;
            objStruct.speed = obj.speed;
            objStruct.font.size = obj.font.size;
            objStruct.font.weight = obj.font.weight; %#ok
            save(sessionFilename,'objStruct','-mat')
        end
        %%
        function load(obj,filename)
            if ~exist(filename,'file'), errordlg2('The file does''n exist');end
            N = length(obj.list);
            for it=fliplr(1:N)
                obj.list{it}.delete;
            end
            load(filename,'-mat');
            if ~exist('objStruct','var'), errordlg2('No session information in this file.');end
            if isfield(objStruct,'startTime'), obj.startTime = objStruct.startTime; else errordlg2('Cannot recreate the session. The file is corrupted.');end
            if isfield(objStruct,'endTime'), obj.endTime = objStruct.endTime; else errordlg2('Cannot recreate the session. The file is corrupted.');end
            if isfield(objStruct,'step'), obj.step = objStruct.step; else errordlg2('Cannot recreate the session. The file is corrupted.');end
            if isfield(objStruct,'nowCursor'), obj.nowCursor = objStruct.nowCursor; else errordlg2('Cannot recreate the session. The file is corrupted.');end
            if isfield(objStruct,'speed'), obj.speed = objStruct.speed; else errordlg2('Cannot recreate the session. The file is corrupted.');end
            if isfield(objStruct,'font')
                obj.font = objStruct.font; 
            else 
                obj.font.size = 12;
                obj.font.weight = 'normal'; % bold
            end
         
            if isfield(objStruct,'list')
                obj.list = []; 
                mobilab = evalin('base','mobilab');
                allDataStreams = mobilab.allStreams;
                N = length(objStruct.list);
                for it=1:N
                    index = allDataStreams.findItem(objStruct.list{it}.uuid);
                    if isempty(index), warndlg2('The session file doesn''t match the content of this folder.');return;end
                    obj.addHandle(allDataStreams.item{index},objStruct.list{it}.browserType,objStruct.list{it});
                end
            else
                obj.list = [];
            end
            
        end
        %%
        function addHandle(obj,dStreamObj,browserType,defaults)
            if nargin < 2, error('Not enough input arguments in addHandle.');end
            if length(dStreamObj.timeStamp) == 1
                error('MoBILAB:noData',['The stream ' dStreamObj.name ' is empty.']);
            end
            if nargin < 3
                switch class(dStreamObj)
                    case 'eeg',          browserType = 'streamBrowser';
                    case 'dataStream',   browserType = 'streamBrowser';
                    case 'mocap',        browserType = 'mocapBrowser';
                    case 'wii',          browserType = 'streamBrowser';
                    case 'audioStream',  browserType = 'audioStream';
                    case 'markerStream', browserType = 'markerStream';
                    case 'videoStream1', browserType = 'videoStream1';
                    case 'videoStream',  browserType = 'videoStream1';
                    case 'sceneStream',  browserType = 'sceneStream';
                    case 'pcaMocap',     browserType = 'projectionBrowser';
                    otherwise,           browserType = 'streamBrowser';
                end
            end
            if strcmp(browserType,'segmentedDataStreamBrowser') || strcmp(browserType,'segmentedMocapBrowser')
                defaults.startTime = max([obj.startTime dStreamObj.originalStreamObj.timeStamp(1)]);
                obj.startTime = defaults.startTime;
                defaults.endTime = min([obj.endTime dStreamObj.originalStreamObj.timeStamp(end)]);
                obj.endTime = defaults.endTime;
            else
                defaults.startTime = max([obj.startTime dStreamObj.timeStamp(1)]);
                obj.startTime = defaults.startTime;
                defaults.endTime = min([obj.endTime dStreamObj.timeStamp(end)]);
                obj.endTime = defaults.endTime;
            end
            defaults.step = obj.step;
            obj.nowCursor = obj.startTime + 2.5;
            defaults.nowCursor = obj.nowCursor;
            defaults.mode = 'slave';
            defaults.font = obj.font;
            
            for it=1:length(obj.list)
                if isa(obj.list{it},'segmentedStreamBrowserHandle')
                    [t1,t2] = obj.list{it}.streamHandle.originalStreamObj.getTimeIndex([obj.startTime obj.endTime]); 
                elseif isa(obj.list{it},'projectionBrowserHandle')
                     [t1,t2] = obj.list{it}.streamHandle.originalStreamObj.getTimeIndex([obj.startTime obj.endTime]); 
                elseif isstruct(obj.list{it}.streamHandle)
                    [~,t1] = min(abs(obj.list{it}.streamHandle.timeStamp-obj.startTime));
                    [~,t2] = min(abs(obj.list{it}.streamHandle.timeStamp-obj.endTime));
                    %t1 = binary_findClosest(obj.list{it}.streamHandle.timeStamp,obj.startTime);
                    %t2 = binary_findClosest(obj.list{it}.streamHandle.timeStamp,obj.endTime);
                else
                    [t1,t2] = obj.list{it}.streamHandle.getTimeIndex([obj.startTime obj.endTime]);
                end
                obj.list{it}.timeIndex = t1:t2;
                obj.list{it}.step = obj.step;
                obj.list{it}.nowCursor = obj.nowCursor;
            end
            
            handles = guidata(obj.master);
            mobilab = handles.mobilab;
            mobilab.lockGui('Making the figure...');
            try
                switch browserType
                    case 'streamBrowser'
                        obj.list{end+1} = DataStreamBrowser(dStreamObj,'slave');
                        obj.bound = max([obj.bound obj.list{end}.windowWidth/2]);
                    case 'mocapBrowser',                  obj.list{end+1} = mocapBrowserHandle(dStreamObj,defaults);
                    case 'videoStream1',                  obj.list{end+1} = videoStreamBrowserHandle1(dStreamObj,defaults);
                    case 'audioStream',                   obj.list{end+1} = audioStreamBrowserHandle(dStreamObj,defaults);
                    case 'markerStream',                  obj.list{end+1} = DataStreamBrowser(dStreamObj,'slave');    
                    case 'sceneStream',                   obj.list{end+1} = sceneBrowserHandle(dStreamObj,defaults);
                    case 'generalizedCoordinatesBrowser', obj.list{end+1} = generalizedCoordinatesBrowserHandle(dStreamObj,defaults);
                    case 'phaseSpaceBrowser',             obj.list{end+1} = phaseSpaceBrowserHandle(dStreamObj,defaults);
                    case 'cometBrowser',                  obj.list{end+1} = cometBrowserHandle(dStreamObj,defaults);
                    case 'segmentedDataStreamBrowser',    obj.list{end+1} = dStreamObj.dataStreamBrowser(defaults);
                    case 'segmentedMocapBrowser',         obj.list{end+1} = dStreamObj.mocapBrowser(defaults);
                    case 'projectionBrowser',             obj.list{end+1} = projectionBrowser(dStreamObj,defaults);
                    case 'vectorBrowser',                 obj.list{end+1} = vectorBrowser(dStreamObj,defaults);
                    otherwise
                        obj.list{end+1} = DataStreamBrowser(dStreamObj,'slave');
                        obj.bound = max([obj.bound obj.list{end}.windowWidth/2]);
                end
            catch ME
                mobilab.lockGui;
                ME.rethrow;
            end
            obj.list{end}.plotThisTimeStamp(obj.nowCursor);
            pos = get(obj.list{end}.figureHandle,'position');
            if length(obj.list) == 1
                set(obj.list{end}.figureHandle,'Position',[0.1 0.5 pos(3:4)]);
                %set(obj.list{end}.figureHandle,'position',[0 1,pos(3:4)]);
            else
                pos2 = get(obj.list{end-1}.figureHandle,'position');
                set(obj.list{end}.figureHandle,'position',[pos2(1)+1.01*pos2(3) pos2(2) pos(3:4)]);
            end
            mobilab.lockGui;
            obj.list{end}.master = obj; 
            set(obj.sliderHandle,'Max',obj.endTime);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.sliderHandle,'Min',obj.startTime);
            set(obj.minTextHandle,'String',num2str(obj.startTime,4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.maxTextHandle,'String',num2str(obj.endTime,4),'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec'],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            
            ed8 = findobj(obj.master,'tag','edit8');
            streamNames = get(ed8,'String');
            if ischar(streamNames), streamNames = {streamNames};end
            streamNames{end+1} = obj.list{end}.streamHandle.name;
            %streamNames = unique(streamNames);
            streamNames(strcmp(streamNames,' ')) = [];
            set(ed8,'Value',1,'String',streamNames);
            
            set( findobj(obj.master,'tag','listbox1'),'Value',1);
            set( findobj(obj.master,'tag','listbox1'),'String',obj.list{end}.streamHandle.event.uniqueLabel);
            obj.plotStep(0);
        end
        %%
        function plotStep(obj,step)
            N = length(obj.list);
            if N, figure(obj.list{1}.figureHandle);end
            for it=1:N, obj.list{it}.plotStep(step);end
            if N
                obj.nowCursor = obj.list{it}.nowCursor;
                set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
                set(obj.sliderHandle,'Value',obj.nowCursor);
            end
            delete(findobj(obj.master,'type','axes'));
        end
        %%
        function plotThisTimeStamp(obj,newNowCursor)
            if newNowCursor > obj.endTime - obj.bound/2
                newNowCursor = obj.endTime - obj.bound/2;
                if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            end
            N = length(obj.list);
            %if N, figure(obj.list{1}.figureHandle);end 
            for it=1:N,
                obj.list{it}.plotThisTimeStamp(newNowCursor);
            end
            if N
                obj.nowCursor = obj.list{it}.nowCursor;
                set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
                set(obj.sliderHandle,'Value',obj.nowCursor);
            end
            delete(findobj(obj.master,'type','axes'));
        end
        function play(obj)
            try
                delete(obj.timerObj);
            end
            fs = 30; % 30 fps should be enough for playback most data types
            obj.playbackStep = obj.speed/(fs);
            obj.timerObj = timer('TimerFcn',{@playCallback, obj}, 'Period', obj.playbackStep,...
                'BusyMode','queue','ExecutionMode','fixedRate','StopFcn',{@stopCallback, obj});
            obj.state = ~obj.state;
        end
        %%
        function delete(obj)
            while ~isempty(obj.list)
                obj.list{1}.delete;
                obj.updateList();
            end
            if ishandle(obj.slHandle), delete(obj.slHandle);end
            if ishandle(obj.ceHandle), delete(obj.ceHandle);end
            if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            delete(obj.timerObj);
            timerObj = obj.timerObj; %#ok
            obj.timerObj = [];
            clear timerObj;
        end
        %%
        function changeSettings(obj)
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            fsize = obj.font.size;
            fweight = obj.font.weight;
            
            prefObj = [...
                PropertyGridField('startTime',obj.startTime,'DisplayName','Start time','Description','')...
                PropertyGridField('endTime',obj.endTime,'DisplayName','Start time','Description','')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('fsize',fsize,'DisplayName','FontSize','Description','')...
                PropertyGridField('fwight',fweight,'Type',PropertyType('char','row',{'normal','bold','light','demi'}),'DisplayName','FontWeight','Description','')...
                PropertyGridField('step', obj.step,'DisplayName','Step','Description','')];
            
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.font.size = val.fsize;
            obj.font.weight = val.fwight;
            
            obj.startTime = val.startTime;
            obj.endTime = val.endTime;
            obj.nowCursor = obj.startTime + obj.bound/2;
            obj.step = val.step;
            
            for it=1:length(obj.list)
                if isa(obj.list{it},'segmentedStreamBrowserHandle') || isa(obj.list{it},'segmentedMocapBrowserHandle') || isa(obj.list{it},'projectionBrowserHandle')
                    [t1,t2] = obj.list{it}.streamHandle.originalStreamObj.getTimeIndex([obj.startTime obj.endTime]);
                else
                    [t1,t2] = obj.list{it}.streamHandle.getTimeIndex([obj.startTime obj.endTime]);
                end
                obj.list{it}.timeIndex = t1:t2;
                obj.list{it}.font = obj.font;
                %set(obj.list{it}.sliderHandle,'Min',obj.startTime);
                %set(obj.list{it}.sliderHandle,'Max',obj.endTime);
                %set( findobj(obj.list{it}.figureHandle,'Tag','text4'),'String', num2str(obj.startTime,4));
                %set( findobj(obj.list{it}.figureHandle,'Tag','text5'),'String', num2str(obj.endTime,4));
            end
            set(obj.sliderHandle,'Min',obj.startTime);
            set(obj.sliderHandle,'Max',obj.endTime);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            obj.plotThisTimeStamp(obj.nowCursor);
            set(obj.timeTexttHandle,'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            set( findobj(obj.master,'Tag','text6'),'String', num2str(obj.startTime,4),'FontSize',obj.font.size,'FontWeight',obj.font.weight)
            set( findobj(obj.master,'Tag','text7'),'String', num2str(obj.endTime,4),'FontSize',obj.font.size,'FontWeight',obj.font.weight)
            set( findobj(obj.master,'Tag','text27'),'FontSize',obj.font.size,'FontWeight',obj.font.weight)
            set( findobj(obj.master,'Tag','text28'),'FontSize',obj.font.size,'FontWeight',obj.font.weight)
        end
        %%
        function updateList(obj)
            N = length(obj.list);
            I = false(N,1);
            for it=1:N
                if ~isvalid(obj.list{it})
                    I(it) = true;
                end
            end
            obj.list(I) = [];
            ed8 = findobj(obj.master,'tag','edit8');
            streamNames = get(ed8,'String');
            if ischar(streamNames), streamNames = {streamNames};end
            streamNames(I) = [];
            if isempty(streamNames), streamNames = ' ';end
            set(ed8,'Value',1,'String',streamNames);
            set( findobj(obj.master,'tag','listbox1'),'Value',1);
            if ~isempty(obj.list)
                set( findobj(obj.master,'tag','listbox1'),'String',obj.list{1}.streamHandle.event.uniqueLabel);
            else
                set( findobj(obj.master,'tag','listbox1'),'String',' ');
            end
        end
        function onMouseWheelMove(obj,~,eventObj)
            try   step = -1*obj.speed*(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/obj.list{1}.streamHandle.samplingRate/2;%#ok
            catch step = -1*obj.speed*(eventObj.VerticalScrollCount*eventObj.VerticalScrollAmount)/512/2;%#ok 
            end
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
        function setTimerPeriod(~,evnt)
            if strcmp(get(evnt.AffectedObject.timerObj,'Running'),'on')
                stop(evnt.AffectedObject.timerObj);
                triggerFlag = true;
            else triggerFlag = false;
            end
            set(evnt.AffectedObject.timerObj,'Period',evnt.AffectedObject.timerMult/evnt.AffectedObject.speed);
            if triggerFlag, start(evnt.AffectedObject.timerObj);end
        end
        %%
        function updateNowCursorDependency(~,evnt)
            if ishandle(evnt.AffectedObject.ceHandle)
                set(findobj(evnt.AffectedObject.ceHandle,'tag','edit1'),'String',num2str(evnt.AffectedObject.nowCursor));
            end
        end
        %%
        function updateListDependency(~,evnt)
            if ishandle(evnt.AffectedObject.slHandle)
                delete(evnt.AffectedObject.slHandle);
                evnt.AffectedObject.slHandle = StreamsList(evnt.AffectedObject);
            end
            if ishandle(evnt.AffectedObject.ceHandle)
                delete(evnt.AffectedObject.ceHandle);
                evnt.AffectedObject.ceHandle = CreateEvent(evnt.AffectedObject);
            end
            if isempty(evnt.AffectedObject.list)
                evnt.AffectedObject.startTime = 0;
                evnt.AffectedObject.endTime = 10000;
            end
        end
        function updateStartTime(~,evnt), set(evnt.AffectedObject.minTextHandle,'String',num2str(evnt.AffectedObject.startTime,4));end
        function updateEndTime(~,evnt),   set(evnt.AffectedObject.maxTextHandle,'String',num2str(evnt.AffectedObject.endTime,4));end
    end
end
%% -----------------
function playCallback(obj, event, bObj) %#ok
plotThisTimeStamp(bObj,bObj.nowCursor+bObj.playbackStep);
drawnow;
end
%%
function stopCallback(obj, event, bObj) %#ok
playObj = findobj(bObj.master,'Tag','play');
CData = get(playObj,'UserData');
if bObj.state, set(playObj,'CData',CData{1});end
stop(bObj.timerObj);
bObj.state = false;
end