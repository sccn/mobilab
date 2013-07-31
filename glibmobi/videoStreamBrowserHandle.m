classdef videoStreamBrowserHandle < browserHandle
    properties
        videoFile
        noNanIndex
        videoFileIndex
        VHandle = [];
        idNumber
        frameRate
        numberOfFrames
        videoCache
    end
    properties(Dependent)
        videoHandle
    end
    methods
        %%
        function obj = videoStreamBrowserHandle(vStreamObj,defaults)
            latBoundary = vStreamObj.event.getLatencyForEventLabel('boundary');
            obj.idNumber = str2double(vStreamObj.event.label);
            obj.noNanIndex = ~isnan(obj.idNumber);
            isEmpty = 0;
            if isempty(vStreamObj.videoFile)
                isEmpty = 1;
                N = length(latBoundary)+1;
                if N
                    videoFile = cell(N,1);%#ok
                    for it=1:N    
                        [FileName,PathName] = uigetfile2({'*.wmv;*.asf;*.asx','Windows Media Video (*.wmv, *.asf, *.asx)';...
                            '*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov','Apple QuickTime?? Movie (*.mov)';...
                            '*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.ogg','Ogg Theora (*.ogg)'},...
                            ['Select the video file number ' num2str(it) ' of ' num2str(N)]);
                        if any([isnumeric(FileName) isnumeric(PathName)]), disp('You must provide a video file.');return;end
                        videoFile{it} = fullfile(PathName,FileName);%#ok
                    end
                    vStreamObj.videoFile = videoFile; %#ok
                end
                vStreamObj.container.save;
            end
            obj.videoFile = vStreamObj.videoFile;
            obj.streamHandle = vStreamObj;
            
            
            
            obj.videoFileIndex = false(length(obj.streamHandle.videoFile),1);
            obj.videoFileIndex(1) = true;
            try
                try
                    N = length(obj.streamHandle.videoFile);
                    hwait = waitbar(1/N,['Reading video 1 of ' num2str(N) '.'],'Color',[0.66 0.76 1]);
                    obj.VHandle = VideoReader(obj.streamHandle.videoFile{1});
                    for it=2:length(obj.streamHandle.videoFile)    
                        waitbar(it/N,hwait,['Reading video ' num2str(it) ' of ' num2str(N) '.']);
                        obj.VHandle(it) = VideoReader(obj.streamHandle.videoFile{it});
                    end
                    close(hwait);
                    obj.frameRate = obj.VHandle(1).FrameRate;
                catch %#ok
                    try close(hwait);end %#ok
                    obj.VHandle = obj.streamHandle.videoFile;
                    mmreadDir = which('mmread');
                    if isempty(mmreadDir)
                        error('mmread toolbox is missing\n. You can download it from:\nhttp://www.mathworks.com/matlabcentral/fileexchange/8028-mmread');
                    end
                    obj.videoCache = mmread(obj.videoHandle,1);
                    warning off %#ok
                    obj.videoCache = mmread(obj.videoHandle,abs(obj.videoCache.nrFramesTotal));
                    it = 1;
                    MaxIter= 10;
                    while isempty(obj.videoCache.frames) && it < MaxIter
                        obj.videoCache = mmread(obj.videoHandle,abs(obj.videoCache.nrFramesTotal));
                        it = it+1;
                    end
                    warning on %#ok
                    obj.videoCache.index = 1;
                    obj.numberOfFrames = abs(obj.videoCache.nrFramesTotal);
                    obj.frameRate = obj.videoCache.rate;
                end
            catch ME
                if ishandle(hwait), close(hwait);end
                if isEmpty
                    vStreamObj.videoFile = '';
                    vStreamObj.saveHeader('f');
                end
                ME.rethrow;
            end
            obj.videoFileIndex(1) = true;
            
            if nargin < 2,
                defaults.startTime = vStreamObj.timeStamp(1);
                defaults.endTime = vStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.mode = 'standalone';
                defaults.speed = 1;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = vStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = vStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = vStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1/obj.frameRate;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            
            obj.uuid = defaults.uuid;
            [t1,t2] = vStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2;
            obj.step = defaults.step;
            obj.nowCursor = vStreamObj.timeStamp(obj.timeIndex(1)) + 2.5;
            obj.onscreenDisplay  = true;
            if isempty(vStreamObj.videoFile), error('You must provide a video file');end
            obj.videoFile = vStreamObj.videoFile;
            
            obj.addlistener('timeIndex','PostSet',@videoStreamBrowserHandle.updateTimeIndexDenpendencies);
            obj.state = false;
            
            obj.noNanIndex = ~isnan(str2double(obj.streamHandle.event.label));
            
            obj.speed = defaults.speed;
            % obj.figureHandle = videoStreamBrowserGUI(obj);
            
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.browserType = 'videoStream';
            defaults.streamName = obj.streamHandle.name;
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            createGraphicObjects@browserHandle(obj);
            plotThisTimeStamp(obj,nowCursor);
        end
        %%
        function videoHandle = get.videoHandle(obj)
            if isa(obj.VHandle(1),'VideoReader')
                videoHandle = obj.VHandle(obj.videoFileIndex);
            else
                if iscell(obj.VHandle)
                    videoHandle = obj.VHandle{obj.videoFileIndex};
                else
                    videoHandle = obj.videoFile;
                end
            end
        end
        %%
        function plotThisTimeStamp(obj,nowCursor,loopFlag)
            if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
                nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            if nargin < 3, loopFlag = false;end
            obj.nowCursor = nowCursor;
            indexList = [];
            if isa(obj.videoHandle,'VideoReader')
                frameIndex = round(interp1(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame(obj.noNanIndex)),obj.idNumber(obj.noNanIndex),obj.nowCursor,'linear','extrap'));
            else
                [~,frameIndex] = min(abs(obj.streamHandle.asfTime{1}-nowCursor));
            end
            
%             if loopFlag
%             
%                 frameIndex = indexList(1);
%             else
%                 frameIndex = round(interp1(obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame(obj.noNanIndex)),obj.idNumber(obj.noNanIndex),obj.nowCursor,'linear','extrap'));
%             end
            
            flags = obj.nowCursor - obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame(~obj.noNanIndex)) > 0;
            if sum(flags) < sum(~obj.noNanIndex) && sum(flags) > 0
                tmp = find(obj.nowCursor - obj.streamHandle.timeStamp(obj.streamHandle.event.latencyInFrame(~obj.noNanIndex)) >0);
                obj.videoFileIndex = false(sum(~obj.noNanIndex),1);
                obj.videoFileIndex(tmp(end)+1) = true;
            elseif sum(flags) == 0
                obj.videoFileIndex = false(sum(~obj.noNanIndex),1);
                obj.videoFileIndex(1) = true;
            else
                obj.videoFileIndex = false(sum(~obj.noNanIndex),1);
                obj.videoFileIndex(end) = true;
            end
            plotFrame(obj,frameIndex);
            % plotFrame2(obj,ind);
            % plotFrame3(obj,ind);
            if obj.onscreenDisplay
                %msg = sprintf('%gsec',obj.nowCursor);
                %fprintf(obj.pipe.fid,'pausing_keep osd_show_text %s\n',msg);
            end
            set(obj.axesHandle,'YTickLabel',[]);
            set(obj.axesHandle,'XTickLabel',[]);
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
            obj.plotThisTimeStamp(newNowCursor,true);
        end
        %%
        function plotFrame2(obj,ind)
            obj.videoCache = mmread(obj.videoFile,ind);
            frame = obj.videoCache.frames.cdata;
            cla(obj.axesHandle);
            image(frame,'Parent',obj.axesHandle);
        end
        %%
        function plotFrame3(obj,ind)
            

            obj.videoCache = mmread(obj.videoFile,ind);
            frame = obj.videoCache.frames.cdata;
            
            cla(obj.axesHandle);
            image(frame,'Parent',obj.axesHandle);
            
        end
        %%
        function plotFrame(obj,frameIndex,indexList)     
            if nargin < 3, indexList = frameIndex;end
            if isempty(indexList), indexList = frameIndex;end
            
            warning off %#ok
            
            frameIndex(frameIndex < 0) = 1;
            frameIndex(frameIndex > obj.numberOfFrames) = obj.numberOfFrames;
                        
            if isa(obj.videoHandle,'VideoReader')
                frame = obj.videoHandle.read(frameIndex);
            else
                tmp = mmread(obj.videoHandle,frameIndex);
                frame = tmp.frames.cdata;
%                 ind = obj.videoCache.index == frameIndex;
%                 if any(ind)
%                     frame = obj.videoCache.frames(ind).cdata;
%                 else
%                     
%                     indexList(indexList > obj.numberOfFrames) = [];
%                     obj.videoCache = [];
%                     obj.videoCache = mmread(obj.videoFile,indexList);
%                     obj.videoCache.index = indexList(1:length(obj.videoCache.frames));
%                     if isempty(obj.videoCache.frames)
%                         tmp = mmread(obj.videoFile,[],[obj.nowCursor-25 obj.nowCursor]);
%                         tmp.frames = tmp.frames(end);
%                         frame = tmp.frames.cdata;
%                         disp('kk')
%                     else
%                         frame = obj.videoCache.frames(1).cdata;
%                         %disp('kk2')
%                     end
%                 end
            end
            
            cla(obj.axesHandle);
            image(frame,'Parent',obj.axesHandle);
            warning on
        end
        %%
        function delete(obj)
            try delete(obj.figureHandle);end %#ok
            if ~strcmp(class(obj.master),'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
            if isa(obj.videoHandle,'VideoReader') && isvalid(obj.videoHandle), obj.videoHandle.delete;end
        end
        %%
        function changeSettings(obj)
            h = videoStreanBrowserSettings(obj);
            uiwait(h);
            try userData = get(h,'userData');catch, userData = [];end%#ok
            try close(h);end %#ok
            if isempty(userData), return;end
            obj.speed = userData.speed;
            obj.onscreenDisplay = userData.onscreenDisplay;
            
            if userData.changeFiles
                N = sum(isnan(obj.idNumber));
                if N
                    for it=1:N
                        [FileName,PathName] = uigetfile2({'*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov',...
                            'Apple QuickTime?? Movie (*.mov)';'*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.wmv;*.asf;*.asx',...
                            'Windows Media Video (*.wmv, *.asf, *.asx)';'*.ogg','Ogg Theora (*.ogg)'},...
                            ['Select the video file number ' num2str(it) ' of ' num2str(N)]);
                        if any([isnumeric(FileName) isnumeric(PathName)]), error('You must provide a video file.');end
                        obj.streamHandle.videoFile{it} = fullfile(PathName,FileName);
                    end
                else
                    [FileName,PathName] = uigetfile2({'*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov',...
                        'Apple QuickTime?? Movie (*.mov)';'*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.wmv;*.asf;*.asx',...
                        'Windows Media Video (*.wmv, *.asf, *.asx)';'*.ogg','Ogg Theora (*.ogg)'},'Select the video file');
                    if any([isnumeric(FileName) isnumeric(PathName)]), error('You must provide a video file.');end
                    obj.streamHandle.videoFile = fullfile(PathName,FileName);
                end
                obj.streamHandle.container.save;
                
                if iscell(obj.streamHandle.videoFile)
                    obj.videoFileIndex = false(length(obj.streamHandle.videoFile),1);
                    hwait = waitdlg([],['Reading the video file 1 of ' num2str(length(obj.streamHandle.videoFile)) '. This can take around 4 min per file.']);
                    obj.VHandle = VideoReader(obj.streamHandle.videoFile{1});
                    close(hwait);
                    for it=2:length(obj.streamHandle.videoFile)
                        hwait = waitdlg([],['Reading the video file ' num2str(it) ' of ' num2str(length(obj.streamHandle.videoFile)) '. This can take around 4 min per file.']);
                        obj.VHandle(it) = VideoReader(obj.streamHandle.videoFile{it});
                        close(hwait);
                    end
                else
                    hwait = waitdlg([],'Reading the video file. This can take around 4 min.');
                    obj.VHandle = VideoReader(obj.streamHandle.videoFile);
                    close(hwait);
                end
                obj.videoFileIndex(1) = true;
            end
            
            figure(obj.figureHandle);
            obj.plotThisTimeStamp(obj.nowCursor);
        end
    end
    %%
    methods(Static)
        function updateTimeIndexDenpendencies(~,evnt)
            if evnt.AffectedObject.timeIndex(1) ~= -1
                evnt.AffectedObject.nowCursor = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)) + 2.5;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
    end
end