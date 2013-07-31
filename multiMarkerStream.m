classdef multiMarkerStream < markerStream
    methods
        function obj = multiMarkerStream(header) 
            if nargin < 1, error('Not enough input arguments.');end
            obj@markerStream(header);
            % obj.addlistener('event','PostSet',@multiMarkerStream.updateDataChannels);
        end
        %%
        function browserObj = plot(obj), browserObj = dataStreamBrowser(obj);end
        function browserObj = dataStreamBrowser(obj,defaults)
            if nargin < 2, defaults.browser = @dataStreamBrowser;end
            descriptor = dir(obj.binFile);
            if descriptor.bytes == 0
                Zeros = zeros(length(obj.timeStamp),1);
                obj.addSamples(Zeros);
                obj.connect;
            end
            defaults.onscreenDisplay = false;
            browserObj = markerStreamBrowserHandle(obj,defaults); 
        end
    end
%     methods(Static)
%         function updateDataChannels(~,evnt)
%             obj = evnt.AffectedObject;
%             uLabel = obj.event.uniqueLabel;
%             obj.numberOfChannels = length(uLabel);
%             data = zeros(size(obj,1),obj.numberOfChannels);
%             fid = fopen(obj.binFile,'w');
%             fwrite(fid,data(:),pbj.precision);
%             fclose(fid);
%             obj.label = uLabel;
%             obj.numberOfChannels;
%             for it=1:obj.numberOfChannels
%                 latency = obj.event.getLatencyForEventLabel(uLabel{it});
%                 indices = obj.getTimeIndex(latency);
%                 obj.mmfObj.Data.x(indices,it) = 1;                
%             end
%         end
%     end
end