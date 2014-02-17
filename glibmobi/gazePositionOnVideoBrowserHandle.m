classdef gazePositionOnVideoBrowserHandle < gazePositionOnScreenBrowserHandle
    properties
        videoFile
        vObj
        imageFlag
    end
    methods
        %% constructor
        function obj = gazePositionOnVideoBrowserHandle(dStreamObj,defaults)
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
                defaults.imageFlag = 0;
            end
            if ~isfield(defaults,'uuid'),        defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'),   defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'),     defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'step'),        defaults.step = 1;end
            if ~isfield(defaults,'mode'),        defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'),       defaults.speed = 1;end
            if ~isfield(defaults,'nowCursor'),   defaults.nowCursor = defaults.startTime + 2.5;end
            if ~isfield(defaults,'font'),        defaults.font = struct('size',12,'weight','normal');end                                       
            if ~isfield(defaults,'imageFlag'),        defaults.imageFlag = 0;end
            
            
            obj@gazePositionOnScreenBrowserHandle(dStreamObj,defaults); 
            
            %if isempty(dStreamObj.videoFile) || ~exist(dStreamObj.videoFile,'file')
                
                if ~defaults.imageFlag
                    [FileName,PathName] = uigetfile2({'*.wmv;*.asf;*.asx','Windows Media Video (*.wmv, *.asf, *.asx)';...
                        '*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov','Apple QuickTime?? Movie (*.mov)';...
                        '*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.ogg','Ogg Theora (*.ogg)'},'Select the video file');
                    if any([isnumeric(FileName) isnumeric(PathName)]), disp('You must provide a video file.');return;end
                    
                else
                    %[FileName,PathName] = uigetfile2({'*.jpg;*.jpeg;*.JPG;*.JPEG','JPEG Image';'*.png','Portable Network Graphics';'*.bmp','Windows Bitmap'},'Select the image file');
                    [FileName,PathName] = uigetfile2('','Select the image file');
                    if any([isnumeric(FileName) isnumeric(PathName)]), disp('No image selected, will use blank background.');return;end
                end
                dStreamObj.videoFile = fullfile(PathName,FileName);
            %end
            
            obj.videoFile = dStreamObj.videoFile;
            try
                if ~defaults.imageFlag
                disp('Reading the video file...')
                obj.vObj = VideoReader(dStreamObj.videoFile);
                else
                    disp('Reading the image file...');
                    obj.vObj = imread(dStreamObj.videoFile);
                end
                disp('Done.')
            catch ME
                ME.rethrow;
            end
            obj.imageFlag = defaults.imageFlag;  
            
            
            
                    
            
            
            
            
            
            %obj.figureHandle = defaults.gui(obj);
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            createGraphicObjects@gazePositionOnScreenBrowserHandle(obj,nowCursor);
            %axis(obj.axesHandle,'off')
            if isempty(obj.videoFile), return;end
             plotThisTimeStamp(obj,nowCursor);
            %createGraphicObjects@videoStreamBrowserHandle1(obj,nowCursor);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            
%             if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
%                 nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
%                 if strcmp(get(obj.timerObj,'Running'),'on')
%                     stop(obj.timerObj);
%                 end
%             end
            
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
            
            obj.nowCursor = nowCursor;
            loc = obj.streamHandle.getTimeIndex(obj.nowCursor);
            plotThisTimeStamp@gazePositionOnScreenBrowserHandle(obj,nowCursor);
            
%             try
%                 if ~obj.imageFlag
%                     frameIndex = obj.streamHandle.data(loc,1);
%                     frameIndex(frameIndex>obj.vObj.NumberOfFrames) = obj.vObj.NumberOfFrames;
%                     frame = obj.vObj.read(frameIndex);
%                 else
%                     frame = obj.vObj;
%                 end
%                 
%                 if size(frame,3) == 1
%                    frame = repmat(frame,[1 1 3]);
%                    obj.vObj = frame;
%                 end
%                 %frame = zeros(size(frame));
%                 [map, cursorLoc] = getHeatMap(obj,nowCursor);
%                 %map = flipud(map);
%                 r1= size(frame,1)/size(map,1); % ratio of images to be accounted for in cursor replacement.
%                 r2= size(frame,2)/size(map,2); %---------------------------
%                 
%                 map = interp1(1:size(map,1),map,linspace(1,size(map,1),size(frame,1)));
%                 map = interp1(1:size(map,2),map',linspace(1,size(map,2),size(frame,2)))';
%                 I = map~=0;
%                 map(I) = map(I)-min(map(I));
%                 map(I) = map(I)./max(map(I));
%                 if any(I(:))
%                 colorIndex = round(map(I)*63)+1;
%                 color = round(jet(64)*254)+1;
%                 value = color(colorIndex,:);
%                 [ind_i,ind_j] = ind2sub(size(map),find(I));
%                 for it=1:length(ind_i), frame(ind_i(it),ind_j(it),:) = value(it,:);end
%                 end
%                 set(obj.gObjHandle,'CData',frame);
%                 set(obj.axesHandle,'xlim',[1 size(frame,2)],'ylim',[1 size(frame,1)]);
%                 %set(obj.cursor,'XData',r2*cursorLoc(1),'YData',...
%                     %r1*cursorLoc(2));
%                 set(obj.axesHandle,'YDir','reverse')
%                 set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
%                 set(obj.sliderHandle,'Value',obj.nowCursor);
%             catch plotThisTimeStamp@gazePositionOnScreenBrowserHandle(obj,nowCursor); %#ok
%             end
            
        end
        %%
        function [map,cursorLoc] = getHeatMap(obj,nowCursor)
            delta = obj.windowWidth/2;
            if  nowCursor + delta < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    nowCursor - delta > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = nowCursor;
            elseif nowCursor + delta >= obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end)) - delta;
                if strcmp(get(obj.timerObj,'Running'),'on'), stop(obj.timerObj);end
            else newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + delta;
            end
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (newNowCursor-obj.windowWidth)));
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (newNowCursor+obj.windowWidth)));
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(newNowCursor-obj.windowWidth/2));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(newNowCursor+obj.windowWidth/2));
            
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1):obj.timeIndex(t2),obj.channelIndex);
            data = data(1:16:end,:);
            data = interp1([1:size(data,1)]',data,[1:1/16:size(data,1)]');
            individualScreenSize = [obj.axesSize(1)/size(data,2)*2 obj.axesSize(2)];
            screenLookings = findWhichScreenIsLookedAt(data);
            
            
            npoints = size(data,1);
            data = [data (1:npoints)'];
            screen = zeros([obj.axesSize npoints]);
            
            notnanind = ~isnan(screenLookings);
            xs = [data(notnanind,end) (screenLookings(notnanind)-1)*2+1];
            ys = [data(notnanind,end) screenLookings(notnanind)*2];
            xsindInData = sub2ind(size(data),xs(:,1),xs(:,2));
            ysindInData = sub2ind(size(data),ys(:,1),ys(:,2));
            
            
            columnLocs = ceil((screenLookings(notnanind)-1)*individualScreenSize(1) + individualScreenSize(1)*data(xsindInData));
            rowLocs  = ceil((1-data(ysindInData))*individualScreenSize(2));
            
            ind = sub2ind(size(screen),columnLocs,rowLocs,data(notnanind,end));
            
            %win = gausswin(size(data,1));
            win = ones(size(data,1),1);
            screen(ind) = 100*win(notnanind);
            
            
            map = mean(screen,3);
            
            map = imfilter(map, obj.kernel);
            map(map==0) = min(map(:));
            map = map - min(map(:));
            if  max(map(:)) ~= 0
                map = map/max(map(:));
            end
            map = map*npoints;
            map = map';
            
            
            t0 = fix(npoints/2);
            if ~isnan(screenLookings(t0))
                [xt,yt] = ind2sub(size(screen(:,:,1)),find(screen(:,:,t0) >= 1));

            else
                xt = -20;
                yt = -20;
            end
            cursorLoc = [xt,yt];
            
        end
        %%
        function obj = changeSettings(obj)
             obj = changeSettings@gazePositionOnScreenBrowserHandle(obj);
%            sg = sign(double(obj.speed<1));
%             sg(sg==0) = -1;
%             speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
%             speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
%             
%              prefObj = [...
%                 PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
%                 PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
%                 PropertyGridField('kernelSize',obj.kernelSize,'DisplayName','Kernel size','Description','Width of the convolution kernel in pixels.')...
%                 PropertyGridField('cursorSize',obj.cursorSize,'DisplayName','Cursor size','Description','Size of the cursor in pixels.')...
%                 ];
%             
%             % create figure
%             f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
%             position = get(f,'position');
%             set(f,'position',[position(1:2) 385 424]);
%             g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
%             uiwait(f); % wait for figure to close
%             val = g.GetPropertyValues();
%             
%             obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
%             obj.windowWidth = val.windowWidth;
%             obj.kernelSize = val.kernelSize;
%             obj.cursorSize = val.cursorSize;
%             figure(obj.figureHandle);
%             obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
%             obj.createGraphicObjects(obj.nowCursor);
%             
%             if isa(obj.master,'browserHandleList')
%                 obj.master.bound = max([obj.master.bound obj.windowWidth]);
%                 obj.master.nowCursor = obj.windowWidth/2;
%                 for it=1:length(obj.master.list)
%                     if obj.master.list{it} ~= obj, obj.master.list{it}.nowCursor = obj.master.nowCursor;end
%                 end
%                 obj.master.plotThisTimeStamp(obj.master.nowCursor);
%             end
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