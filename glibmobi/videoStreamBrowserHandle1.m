classdef videoStreamBrowserHandle1 < browserHandle
    properties
        videoFile
        vObj
        img
        Frames = [];
    end
    methods
        %%
        function obj = videoStreamBrowserHandle1(vStreamObj,defaults)
            
            if isempty(vStreamObj.videoFile) || ~exist(vStreamObj.videoFile,'file')
                formats = VideoReader.getFileFormats();
                [FileName,PathName] = uigetfile2(getFilterSpec(formats),'Select the video file');
                if any([isnumeric(FileName) isnumeric(PathName)]), disp('You must provide a video file.');return;end
                vStreamObj.videoFile = fullfile(PathName,FileName);
            end
            
            obj.videoFile = vStreamObj.videoFile;
            obj.streamHandle = vStreamObj;
            
            try
                obj.vObj = VideoReader(obj.streamHandle.videoFile);
            catch ME
                ME.rethrow;
            end
            
            if nargin < 2,
                defaults.startTime = vStreamObj.timeStamp(1);
                defaults.endTime = vStreamObj.timeStamp(end);
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.step = 1;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = vStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = vStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = vStreamObj.timeStamp(end);end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            
            obj.uuid = defaults.uuid;
            [t1,t2] = vStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2;
            obj.nowCursor = vStreamObj.timeStamp(obj.timeIndex(1)) + 2.5;
            obj.onscreenDisplay  = true;
            if isempty(vStreamObj.videoFile), error('You must provide a video file');end
            obj.videoFile = vStreamObj.videoFile;
            try
                disp('Reading the video file...')
                obj.Frames = obj.vObj.read();
                disp('Done.')
            end
            obj.state = false;
            obj.step = defaults.step;
            obj.addlistener('timeIndex','PostSet',@videoStreamBrowserHandle1.updateTimeIndexDenpendencies);
            obj.speed = defaults.speed;
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
        function plotThisTimeStamp(obj,nowCursor)
            if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
                nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            obj.nowCursor = nowCursor;
            loc = obj.streamHandle.getTimeIndex(obj.nowCursor);
            frameIndex = obj.streamHandle.data(loc);    
            frameIndex(frameIndex>obj.vObj.NumberOfFrames) = obj.vObj.NumberOfFrames;
            if ~isempty(obj.Frames)
                frame = obj.Frames(:,:,:,frameIndex);
            else
                frame = obj.vObj.read(frameIndex);
            end
            if isempty(obj.img)
                cla(obj.axesHandle);
                obj.img = image(frame,'Parent',obj.axesHandle);
            else
                set(obj.img,'CData', frame);
            end
            %warning on %#ok
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
            try delete(obj.figureHandle);end %#ok
            if ~isa(obj.master,'browserHandleList')
                try delete(obj.master);end %#ok
            else
                obj.timeIndex = -1;
                obj.master.updateList;
            end
            if isvalid(obj.vObj), delete(obj.vObj);end
        end
        %%
        function changeSettings(obj)
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
             prefObj = [...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('videoFile',obj.streamHandle.videoFile,'DisplayName','Video file','Description','The following formats are supported: .avi, .mpg, .mov, .mp4, .m4v .wmv, .asf, .asx, .wmv, .asf, .asx, and .ogg')...
                ];
            
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            try
                if ~strcmp(val.videoFile,obj.streamHandle.videoFile)
                    obj.streamHandle.videoFile = val.videoFile;
                    obj.vObj = VideoReader(obj.streamHandle.videoFile);
                end
            end
%             [FileName,PathName] = uigetfile2({'*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov',...
%                 'Apple QuickTime?? Movie (*.mov)';'*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.wmv;*.asf;*.asx',...
%                 'Windows Media Video (*.wmv, *.asf, *.asx)';'*.ogg','Ogg Theora (*.ogg)'},'Select the video file');
%             if any([isnumeric(FileName) isnumeric(PathName)]),
%                 disp('No file changed.');
%                 return;
%             end
%             obj.streamHandle.videoFile = fullfile(PathName,FileName);
%             obj.vObj = VideoReader(obj.streamHandle.videoFile);
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