classdef segmentedMocapBrowserHandle < mocapBrowserHandle
     properties
         lightColor = [0.9412 0.9412 0.9412];
         segmentObj
         nameSegmentedObject
     end
    methods
        %% constructor
        function obj = segmentedMocapBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.originalStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.originalStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.showChannelNumber = false;
                defaults.channels = 1:dStreamObj.numberOfChannels/3;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.floorColor = [0.5294 0.6118 0.8706];
                defaults.background = [0 0 0.3059];
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.originalStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.originalStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'showNumberFlag'), defaults.showNumberFlag = 0;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels/3;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'floorColor'), defaults.floorColor = [0.5294 0.6118 0.8706];end
            if ~isfield(defaults,'background'), defaults.background = [0 0 0.3059];end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 2.5;end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end

            obj@mocapBrowserHandle(dStreamObj.originalStreamObj,defaults);
            obj.uuid = defaults.uuid;
            obj.nameSegmentedObject = dStreamObj.name;
            obj.segmentObj = dStreamObj.segmentObj;
            obj.plotThisTimeStamp(obj.nowCursor);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@mocapBrowserHandle(obj);
            defaults.browserType = 'segmentedMocapBrowser';
            defaults.streamName = obj.nameSegmentedObject;
        end
        %% plot
        function createGraphicObjects(obj,nowCursor)
            createGraphicObjects@mocapBrowserHandle(obj,nowCursor); 
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            plotThisTimeStamp@mocapBrowserHandle(obj,nowCursor);
            ind = nowCursor > obj.segmentObj.startLatency & nowCursor < obj.segmentObj.endLatency;
            if any(ind), 
                set(obj.markerHandle,{'MarkerFaceColor'},num2cell(obj.color,2));
                try set(obj.lineHandle,'Color',obj.lineColor,'LineStyle','-');end %#ok
            else
                set(obj.markerHandle,'MarkerFaceColor',obj.lightColor);
                try set(obj.lineHandle,'Color',obj.lightColor,'LineStyle','-.');end
            end
        end
    end
end