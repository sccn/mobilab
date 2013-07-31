classdef roi < handle
    properties
        region
    end
    methods
        function obj = roi()
        end
    end
    methods(Abstract)
        mObj = applyROI(obj,streamObj)
        paint(obj,axesHandle)
    end
end