classdef timeSeriesRoi < roi
    properties
        segmentObj
        rectHandle
        axesHandle;
        isactive = false
        color = [0.93 0.96 1];
    end
    properties(GetAccess=private,SetAccess=private)
        axesHandleOri;
    end
    methods
        function obj = timeSeriesRoi(aHandle)
            obj.segmentObj = basicSegment(zeros(1,2),'roi');
            obj.axesHandleOri = aHandle;
            obj.axesHandle = copyobj(obj.axesHandleOri,get(aHandle,'Parent'));
            cla(obj.axesHandle);
            set(obj.axesHandle,'Visible','off');
            set(obj.axesHandle,'XTick',[],'YTick',[],'XTickLabel','','YTickLabel','');
        end
        %%
        function paint(obj)
            if ~obj.isactive, cla(obj.axesHandle);return;end
            x_lim = get(obj.axesHandleOri,'Xlim');
            y_lim = get(obj.axesHandleOri,'Ylim');
            N = 512;
            time = linspace(x_lim(1),x_lim(2),N)';
            I = ismember(obj,time);
            cla(obj.axesHandle);
            if isempty(I), return;end
            set(obj.axesHandle,'Xlim',x_lim,'Ylim',y_lim);
            
            for it=1:length(obj.segmentObj.startLatency)
                try %#ok
                    pointer1(1) = time(find(time > obj.segmentObj.startLatency(it),1,'first'));
                    pointer2(1) = time(find(time < obj.segmentObj.endLatency(it),1,'last'));
                    h = patch([pointer1 pointer2 pointer2 pointer1],[y_lim([1 1]) y_lim([2 2])],'g',...
                        'Parent',obj.axesHandle,'FaceColor',obj.color,'EdgeColor',obj.color,'FaceAlpha',0.75,'UserData',it);
                    set(h,'ButtonDownFcn',@(src, event)deleteSegment(obj,h,event),'Tag','rectangle')
                end
            end
            drawnow;
        end
        %%
        function deleteSegment(obj,rectHandle,~)
            browserObj = get(get(obj.axesHandle,'parent'),'userData');
            if strcmp(browserObj.zoomHandle.Enable,'on'), return;end
                
            index = get(rectHandle,'userData');
            h = sort(findobj(get(rectHandle,'parent'),'tag','rectangle'));
            userData = get(h,'userData');
            if iscell(userData)
                ind = cell2mat(userData);
            else
                ind = userData;
            end
            ind(ind>index) = ind(ind>index)-1;
            set(h,{'userData'},num2cell(ind));
            
            obj.segmentObj.startLatency(index) = [];
            obj.segmentObj.endLatency(index) = [];
            delete(rectHandle);
        end
        %%
        function applyROI(obj,streamObj)
        end
        %% 
        function addSegment(obj,segObj)
            if isempty(obj.segmentObj.startLatency) || all([obj.segmentObj.startLatency obj.segmentObj.endLatency] == 0)
                obj.segmentObj = segObj;
                return;
            end
            try
            obj.segmentObj = cat(obj.segmentObj,segObj);
            if all([~obj.segmentObj.startLatency(1) ~obj.segmentObj.endLatency(1)])
                obj.segmentObj.startLatency(1) = [];obj.segmentObj.endLatency(1) = [];
            end
            obj.segmentObj.segmentName = 'roi';
            catch ME
                ME.rethrow;
            end 
        end
        %%
        function I = ismember(obj,timeVector)
            I = false(length(timeVector),1);
            N = length(obj.segmentObj.startLatency);
            for it=1:N
                I = I | (timeVector >= obj.segmentObj.startLatency(it) & timeVector < obj.segmentObj.endLatency(it));
            end
            %I = find(I);
        end
    end
end