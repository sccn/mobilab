classdef gazeStreamBrowserHandle < mocapBrowserHandle
    properties
        gazeHandle
        pupilHandle
        gazeStreamObj
        gazeLineColor = [0.7294 0.8314 0.9569];
    end
    methods
        function obj = gazeStreamBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels/3;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.floorColor = [0.5294 0.6118 0.8706];
                defaults.background = [0 0 0.3059];
                defaults.font = struct('size',12,'weight','normal');
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'showNumberFlag'), defaults.showNumberFlag = 0;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels/3;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'floorColor'), defaults.floorColor = [0.5294 0.6118 0.8706];end
            if ~isfield(defaults,'background'), defaults.background = [0 0 0.3059];end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
           
            mocapObj = [];
            for it=fliplr(1:length(dStreamObj.container.item))
                if isa(dStreamObj.container.item{it},'mocap')
                    mocapObj = dStreamObj.container.item{it};
                end
            end
            if isempty(mocapObj), error('Could not find the mocap data.');end
            obj@mocapBrowserHandle(mocapObj,defaults);
            obj.gazeStreamObj = dStreamObj;
            obj.uuid = defaults.uuid;
            
            %[~,t0] = min(abs(dStreamObj.timeStamp - obj.nowCursor));
            t0 = binary_findClosest(dStreamObj.timeStamp, obj.nowCursor);
            hold(obj.axesHandle,'on');
            x = [obj.gazeStreamObj.eyePosition(t0,1) obj.gazeStreamObj.gazePosition(t0,1)];
            y = [obj.gazeStreamObj.eyePosition(t0,2) obj.gazeStreamObj.gazePosition(t0,2)];
            z = [obj.gazeStreamObj.eyePosition(t0,3) obj.gazeStreamObj.gazePosition(t0,3)];
            obj.gazeHandle = line(x,y,z,'Parent',obj.axesHandle,'Color',obj.gazeLineColor);
            hold(obj.axesHandle,'off')
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            createGraphicObjects@mocapBrowserHandle(obj,nowCursor);
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            plotThisTimeStamp@mocapBrowserHandle(obj,nowCursor);
            %[~,t0] = min(abs(obj.gazeStreamObj.timeStamp - obj.nowCursor));
            t0 = binary_findClosest(obj.gazeStreamObj.timeStamp,obj.nowCursor);
            x = [obj.gazeStreamObj.eyePosition(t0,1) obj.gazeStreamObj.gazePosition(t0,1)];
            y = [obj.gazeStreamObj.eyePosition(t0,2) obj.gazeStreamObj.gazePosition(t0,2)];
            z = [obj.gazeStreamObj.eyePosition(t0,3) obj.gazeStreamObj.gazePosition(t0,3)];
            set(obj.gazeHandle,'XData',x,'YData',y,'ZData',z);
        end
    end
end