classdef audioStreamBrowserHandle < streamBrowserHandle
    methods
        %% constructor
        function obj = audioStreamBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.gain = 0.25;
                defaults.normalizeFlag = false;
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
                defaults.font = struct('size',12,'weight','normal');
                defaults.gui = @streamBrowserNG;
                defaults.onscreenDisplay = true;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'gain'), defaults.gain = 0.25;end
            if ~isfield(defaults,'normalizeFlag'), defaults.normalizeFlag = false;end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'showChannelNumber'), defaults.showChannelNumber = false;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if defaults.windowWidth > defaults.endTime - defaults.startTime, defaults.windowWidth = defaults.endTime - defaults.startTime; end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'gui'), defaults.gui = @streamBrowserNG;end
            if ~isfield(defaults,'onscreenDisplay'), defaults.onscreenDisplay = true;end
            
            obj@streamBrowserHandle(dStreamObj,defaults);
            hb  = findobj(obj.figureHandle,'tag','settings');
            hbc = copyobj(hb,get(hb,'parent'));
            
            pos = get(hb,'position');
            set(hbc,'position',[pos(1)*1.35 pos(2:4)])
            
            mobilabPath = dStreamObj.container.container.path;
            icon = [mobilabPath filesep 'skin' filesep '32px-Gnome-audio-volume-high.svg.png'];
            CData = imread(icon);
            set(hbc,'CData',CData);
            
            set(hbc,'Callback',@(src)playSound(obj));
            set(hbc,'Callback',@(src,event)playSound(obj,[],event));
        end
        function playSound(obj,~,~)
            [~,t1] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor-obj.windowWidth/2)));  
            [~,t2] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - (obj.nowCursor+obj.windowWidth/2)));  
            data = obj.streamHandle.data(obj.timeIndex(t1:t2),obj.channelIndex);
            if size(data,2)>2
                for it=1:size(data,2)
                    sound(data(:,it),obj.streamHandle.samplingRate);
                end
            else
                sound(data,obj.streamHandle.samplingRate);
            end
        end
    end
end