classdef gazePositionOnScreenBrowserHandle < browserHandle
    properties
        gObjHandle
        windowWidth
        axesSize
        channelIndex
        cursor
        cursorSize
        markerSize
        videoFile
        vObj
        imageFlag
    end
    methods
        %% constructor
        function obj = gazePositionOnScreenBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.windowWidth = 2;  % 2 seconds;
                defaults.showChannelNumber = false;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + 2.5;
                defaults.font = struct('size',12,'weight','normal');
                defaults.imageFlag = 0;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 2;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + 2.5;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end  
            if ~isfield(defaults,'imageFlag'),        defaults.imageFlag = 0;end
            
            
            defaults.channels = 1:dStreamObj.numberOfChannels;
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
            obj.markerSize = 100;
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            %obj.timeStamp = obj.streamHandle.timeStamp(obj.timeIndex);
            obj.step = defaults.step;       % half a second
            obj.windowWidth = defaults.windowWidth;
            obj.nowCursor = defaults.nowCursor;
            obj.channelIndex = defaults.channels;
            numberOfScreens = length(defaults.channels)/2;
            %obj.axesSize = ceil(prctile(dStreamObj.data(:,defaults.channels),95));
            obj.axesSize = [numberOfScreens*600 1000]; %this should be an input 
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end;
            
            
             %if isempty(dStreamObj.videoFile) || ~exist(dStreamObj.videoFile,'file')
                
%                 if ~defaults.imageFlag
%                     [FileName,PathName] = uigetfile2({'*.wmv;*.asf;*.asx','Windows Media Video (*.wmv, *.asf, *.asx)';...
%                         '*.avi','AVI (*.avi)';'*.mpg','MPEG-1 (*.mpg)';'*.mov','Apple QuickTime?? Movie (*.mov)';...
%                         '*.mp4;*.m4v','MPEG-4 Mac (*.mp4, *.m4v)';'*.ogg','Ogg Theora (*.ogg)'},'Select the video file');
%                     if any([isnumeric(FileName) isnumeric(PathName)]), disp('You must provide a video file.');return;end
%                     
%                 else
                    %[FileName,PathName] = uigetfile2({'*.jpg;*.jpeg;*.JPG;*.JPEG','JPEG Image';'*.png','Portable Network Graphics';'*.bmp','Windows Bitmap'},'Select the image file');
                    [FileName,PathName] = uigetfile2('','Select the image file');
                    
                    if any([isnumeric(FileName) isnumeric(PathName)]),
                        disp('No image selected, will use blank background.');
                    else
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
                    end
                    %                 end
                
            obj.imageFlag = defaults.imageFlag;
            obj.figureHandle = defaults.gui(obj);    
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.axesHandle,'drawmode','fast');
%             
%             %obj.kernel = fspecial('gaussian',obj.kernelSize,obj.kernelSize);
%             obj.kernel = fspecial('disk',obj.kernelSize);
%             %obj.kernel = fspecial('log',obj.kernelSize,100);
                                    
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            %[~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            
            
            
            %%%% we only plot the data [-obj.WindowWidth,0] sec. i.e, we don't plot the future eye data. 
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor));
            data = obj.streamHandle.data(obj.timeIndex(t1):obj.timeIndex(t2)-1,obj.channelIndex);
            
            numberOfScreens = size(data,2)/2;
            individualScreenSize = [obj.axesSize(1)/numberOfScreens obj.axesSize(2)];
            screenLookings = findWhichScreenIsLookedAt(data); %if no screen is looked at, the value is NaN.
            
            npoints = size(data,1);
            data = [data (1:npoints)'];
            
            notnanind = ~isnan(screenLookings);
            xs = [data(notnanind,end) (screenLookings(notnanind)-1)*2+1];
            ys = [data(notnanind,end) screenLookings(notnanind)*2];
            xsindInData = sub2ind(size(data),xs(:,1),xs(:,2));
            ysindInData = sub2ind(size(data),ys(:,1),ys(:,2));
            
            
            
            %this -5000 is to plot NaN points outside the axes. 
            %If they are not plotted either via excluding in scatter or with NaN's, 
            %it is hard (maybe impossible) to arrange marker colors accordingly.
            columnLocs = -5000*ones(size(data,1),1); 
            rowLocs = -5000*ones(size(data,1),1);
            columnLocs(notnanind) = ceil((screenLookings(notnanind)-1)*individualScreenSize(1) + individualScreenSize(1)*data(xsindInData));
            rowLocs(notnanind)  = ceil((data(ysindInData))*individualScreenSize(2));
            %-----------------------------------------------------------
            
            
            cla(obj.axesHandle);
            
            % Plots the background image.
            if obj.imageFlag
                imagesc([0 obj.axesSize(1)],[0 obj.axesSize(2)],flipdim(im2double(obj.vObj),1)); 
            end
            %----------------------
            
            hold(obj.axesHandle,'on')
            colordata = linspace(0,1,length(columnLocs));
            obj.gObjHandle(1) = scatter(obj.axesHandle,columnLocs,rowLocs,obj.markerSize,colordata');
            
            set(obj.gObjHandle(1),'linewidth',2.5);
            
            
            % connecting consecutive data points
            if ~any(notnanind)
                obj.gObjHandle(2) = plot(0,0);
            else
                obj.gObjHandle(2) = plot(columnLocs(notnanind),rowLocs(notnanind));
            end
            %---------------------------
            
            % Strings to label strings
            ScreenLabels{numberOfScreens} = ['Screen ' num2str(numberOfScreens)];
            for i = 1:numberOfScreens-1
                tmp = line([i*individualScreenSize(1) i*individualScreenSize(1) ],[0  individualScreenSize(2)]);
                set(tmp,'color','r');
                set(tmp,'linestyle','- -')
                ScreenLabels{i} = ['Screen ' num2str(i)]; 
            end
            %-----------------------------------
            
            hold(obj.axesHandle,'off')
            title(obj.axesHandle,['Blue Circle: -' num2str(obj.windowWidth) ' sec. ' 'Red circle: Current time point']);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor) ' sec']);
            
            set(obj.axesHandle,'xlim',[0 obj.axesSize(1)])
            set(obj.axesHandle,'ylim',[0 obj.axesSize(2)])
            set(obj.axesHandle,'yticklabel','');
            set(obj.axesHandle,'xtick',linspace(obj.axesSize(1)/numberOfScreens/2,obj.axesSize(1)-obj.axesSize(1)/numberOfScreens/2,numberOfScreens));
            set(obj.axesHandle,'xticklabel',ScreenLabels);
            set(obj.axesHandle,'ydir','normal')
            
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
            
            %%%% we only plot the data [-obj.WindowWidth,0] sec. i.e, we don't plot the future eye data. 
            t1 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor-obj.windowWidth));
            t2 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex),(obj.nowCursor));
            
            if t1==t2, return;end
            
            data = obj.streamHandle.mmfObj.Data.x(obj.timeIndex(t1):obj.timeIndex(t2)-1,obj.channelIndex);
            data = data(1:16:end,:);
            %data = interp1([1:size(data,1)]',data,[1:1/16:size(data,1)]');
            
            individualScreenSize = [obj.axesSize(1)/size(data,2)*2 obj.axesSize(2)];
            screenLookings = findWhichScreenIsLookedAt(data);
            
            npoints = size(data,1);
            data = [data (1:npoints)'];
            
            
            notnanind = ~isnan(screenLookings);
            xs = [data(notnanind,end) (screenLookings(notnanind)-1)*2+1];
            ys = [data(notnanind,end) screenLookings(notnanind)*2];
            xsindInData = sub2ind(size(data),xs(:,1),xs(:,2));
            ysindInData = sub2ind(size(data),ys(:,1),ys(:,2));
            
            %this -5000 is to plot NaN points outside the axes. 
            %If they are not plotted either via excluding in scatter or with NaN's, 
            %it is hard (maybe impossible) to arrange marker colors accordingly.
            columnLocs = -5000*ones(size(data,1),1); 
            rowLocs = -5000*ones(size(data,1),1);
            columnLocs(notnanind) = ceil((screenLookings(notnanind)-1)*individualScreenSize(1) + individualScreenSize(1)*data(xsindInData));
            rowLocs(notnanind)  = ceil((data(ysindInData))*individualScreenSize(2));
            %-------------------------------------------------------------
            
            colordata = linspace(0,1,length(columnLocs));
            
            % data points updated
            set(obj.gObjHandle(1),'xdata',columnLocs','ydata',rowLocs','cdata',colordata);
            %--------------------------------------------
            hold(obj.axesHandle,'on')
            
            % lines between data points updated
            set(obj.gObjHandle(2),'xdata',columnLocs(notnanind),'ydata',rowLocs(notnanind));
            %-----------------------------------
            hold(obj.axesHandle,'off')
           
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
            set(obj.axesHandle,'xlim',[0 obj.axesSize(1)])
            set(obj.axesHandle,'ylim',[0 obj.axesSize(2)]);
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
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5 20];
            
             prefObj = [...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x','20x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('windowWidth',obj.windowWidth,'DisplayName','Window width','Description','Size of the segment to plot in seconds.')...
                PropertyGridField('cursorSize',obj.cursorSize,'DisplayName','Cursor size','Description','Size of the cursor in pixels.')...
                PropertyGridField('markerSize',obj.markerSize,'DisplayName','Marker size','Description','Size of the markers that show gaze locations.')...
                PropertyGridField('goToTime',obj.nowCursor,'DisplayName','Go To Time','Description','Go to this time frame.')...
                ];
            
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
            
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x','20x'},val.speed));
            obj.windowWidth = val.windowWidth;
            obj.cursorSize = val.cursorSize;
            obj.markerSize = val.markerSize;
            obj.nowCursor = val.goToTime;
            figure(obj.figureHandle);
            %obj.nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1)) + obj.windowWidth/2;
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