classdef pcdBrowserHandle < browserHandle
    properties
        hLabels
        hSensors
        hScalp
        hCortex
        hVector
        dcmHandle
        pointer
        interpolator
        numberOfPoints
        sourceMagnitud
        sourceOrientation
        scalpData
        surfData
    end
    methods
        %% constructor
        function obj = pcdBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.windowWidth = 5;  % 5 seconds;
                defaults.mode = 'standalone';
                defaults.speed = 1;
                defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;
                defaults.font = struct('size',12,'weight','normal');
                defaults.gui = @streamBrowserNG;
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'windowWidth'), defaults.windowWidth = 5;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'nowCursor'), defaults.nowCursor = defaults.startTime + defaults.windowWidth/2;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'gui'), defaults.gui = @streamBrowserNG;end
            
            obj.streamHandle = dStreamObj;
            obj.timeIndex = 1:size(dStreamObj,2);
            obj.uuid         = defaults.uuid;
            obj.font         = defaults.font;
            obj.speed        = defaults.speed;
            obj.state        = false;
            obj.step         = defaults.step;       % half a second
            obj.nowCursor    = defaults.nowCursor;

            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            obj.figureHandle = defaults.gui(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.browserType = 'pcdBrowser';
        end
        %%
        function createGraphicObjects(obj,nowCursor)
            
            createGraphicObjects@browserHandle(obj);
            set(obj.figureHandle,'Renderer','opengl')
            
            load(obj.streamHandle.surfaces);
            obj.surfData = surfData;
            mPath = obj.streamHandle.container.container.path;
            path = fullfile(mPath,'skin');
            labelsOn  = imread([path filesep 'labelsOn.png']);
            labelsOff = imread([path filesep 'labelsOff.png']);
            sensorsOn = imread([path filesep 'sensorsOn.png']);
            sensorsOff = imread([path filesep 'sensorsOff.png']);
            scalpOn = imread([path filesep 'scalpOn.png']);
            scalpOff = imread([path filesep 'scalpOff.png']);
            vectorOn = imread([path filesep 'vectorOn.png']);
            vectorOff = imread([path filesep 'vectorOff.png']);
            
            toolbarHandle = findall(obj.figureHandle,'Type','uitoolbar');
            
            hcb(1) = uitoggletool(toolbarHandle,'CData',labelsOff,'Separator','on','HandleVisibility','off','TooltipString','Labels On/Off','userData',{labelsOn,labelsOff},'State','off');
            set(hcb(1),'OnCallback',@(src,event)rePaint(obj,hcb(1),'labelsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(1),'labelsOff'));
            
            hcb(2) = uitoggletool(toolbarHandle,'CData',sensorsOff,'Separator','off','HandleVisibility','off','TooltipString','Sensors On/Off','userData',{sensorsOn,sensorsOff},'State','off');
            set(hcb(2),'OnCallback',@(src,event)rePaint(obj,hcb(2),'sensorsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(2),'sensorsOff'));
            
            hcb(3) = uitoggletool(toolbarHandle,'CData',scalpOff,'Separator','off','HandleVisibility','off','TooltipString','Scalp On/Off','userData',{scalpOn,scalpOff},'State','off');
            set(hcb(3),'OnCallback',@(src,event)rePaint(obj,hcb(3),'scalpOn'),'OffCallback',@(src, event)rePaint(obj,hcb(3),'scalpOff'));
            
            hcb(4) = uitoggletool(toolbarHandle,'CData',vectorOff,'Separator','off','HandleVisibility','off','TooltipString','Orientation On/Off','userData',{vectorOn,vectorOff},'State','off');
            set(hcb(4),'OnCallback',@(src,event)rePaint(obj,hcb(4),'vectorOn'),'OffCallback',@(src, event)rePaint(obj,hcb(4),'vectorOff'));
            
            obj.dcmHandle = datacursormode(obj.figureHandle);
            obj.dcmHandle.SnapToDataVertex = 'off';
            set(obj.dcmHandle,'UpdateFcn',@(src,event)showLabel(obj,event));
            obj.dcmHandle.Enable = 'off';
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t] = min(abs(obj.streamHandle.timeStamp - obj.nowCursor));
            t = binary_findClosest(obj.streamHandle.timeStamp, obj.nowCursor); 
            J = obj.streamHandle.mmfObj.Data.x(:,t);
            
            cla(obj.axesHandle);
            hold(obj.axesHandle,'on');
            
            obj.hSensors = scatter3(obj.axesHandle,obj.streamHandle.parent.channelSpace(:,1),obj.streamHandle.channelSpace(:,2),...
                obj.streamHandle.channelSpace(:,3),'filled','MarkerEdgeColor','k','MarkerFaceColor','y');
            set(obj.hSensors,'Visible','off');
            
            N = length(obj.streamHandle.parent.label);
            k = 1.1;
            obj.hLabels = zeros(N,1);
            for it=1:N, obj.hLabels(it) = text('Position',k*obj.streamHandle.parent.channelSpace(it,:),'String',obj.streamHandle.parent.label{it},'Parent',obj.axesHandle);end
            set(obj.hLabels,'Visible','off');
                        
            obj.numberOfPoints = size(obj.surfData(3).vertices,1);
            if size(J,1) == 3*obj.numberOfPoints
                J = reshape(J,[size(J,1)/3 3 size(J,2)]);
                Jm = squeeze(sqrt(sum(J.^2,2)));
                normals = J;
            else
                Jm = J;
                normals = geometricTools.getSurfaceNormals(obj.surfData(3).vertices,obj.surfData(3).faces,false);    
            end
            obj.sourceMagnitud = Jm;
            obj.sourceOrientation = J;
            
            % vectors
            obj.hVector = quiver3(obj.axesHandle,obj.surfData(3).vertices(:,1),obj.surfData(3).vertices(:,2),obj.surfData(3).vertices(:,3),normals(:,1,1),normals(:,2,1),normals(:,3,1),2);
            set(obj.hVector,'Color','k','Visible','off');
           
             % cortex
            obj.hCortex = patch('vertices',obj.surfData(3).vertices,'faces',obj.surfData(3).faces,'FaceVertexCData',obj.sourceMagnitud,...
                'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',1,'SpecularColorReflectance',0,...
                'SpecularExponent',50,'SpecularStrength',0.5,'Parent',obj.axesHandle);
            camlight(0,180)
            camlight(0,0)
            
%             l = size(obj.streamHandle,2);
%             try tmp = obj.streamHandle.mmfObj.Data.x(:,1:10*obj.streamHandle.samplingRate);
%                 tmp = [tmp obj.streamHandle.mmfObj.Data.x(:,fix(l/2): fix(l/2)+10*obj.streamHandle.samplingRate)];
%                 tmp = [tmp obj.streamHandle.mmfObj.Data.x(:,end-10*obj.streamHandle.samplingRate:end)];
%             catch tmp = obj.streamHandle.mmfObj.Data.x;
%             end
%             mx = prctile(abs(tmp(:)),95);
%             mx(mx==0) = max(abs(tmp(:)));
            % scalp
            obj.interpolator = geometricTools.localGaussianInterpolator(obj.streamHandle.channelSpace,obj.surfData(1).vertices,6);
            obj.scalpData = obj.interpolator*obj.streamHandle.parent.mmfObj.Data.x(t,:)';
            obj.hScalp = patch('vertices',obj.surfData(1).vertices,'faces',obj.surfData(1).faces,'FaceVertexCData',obj.scalpData,...
                'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',0.85,'SpecularColorReflectance',0,...
                'SpecularExponent',50,'SpecularStrength',0.5,'Parent',obj.axesHandle,'Visible','off');
            
            colormap(bipolar(512, 0.99))
            % set(obj.axesHandle,'Clim',[-mx mx]);
            axis(obj.axesHandle,'equal');
            axis(obj.axesHandle,'vis3d');
            axis(obj.axesHandle,'off')
            rotate3d
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
        end
        %%
        function plotThisTimeStamp(obj,nowCursor)
            if nowCursor > obj.streamHandle.timeStamp(end)
                nowCursor = obj.streamHandle.timeStamp(end);
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t] = min(abs(obj.streamHandle.timeStamp - obj.nowCursor));
            t = binary_findClosest(obj.streamHandle.timeStamp, obj.nowCursor);
            J = obj.streamHandle.mmfObj.Data.x(:,t);
            
            if size(J,1) == 3*obj.numberOfPoints
                J = reshape(J,[size(J,1)/3 3 size(J,2)]);
                Jm = squeeze(sqrt(sum(J.^2,2)));
                normals = J;
            else
                Jm = J;
                normals = geometricTools.getSurfaceNormals(obj.surfData(3).vertices,obj.surfData(3).faces,false);    
            end
            obj.sourceMagnitud = Jm;
            obj.sourceOrientation = normals;
            obj.scalpData = obj.interpolator*obj.streamHandle.parent.mmfObj.Data.x(t,:)';
                        
            set(obj.hCortex,'FaceVertexCData',obj.sourceMagnitud);
            set(obj.hVector,'UData',obj.sourceOrientation(:,1),'VData',obj.sourceOrientation(:,2),'WData',obj.sourceOrientation(:,3));
            set(obj.hScalp,'FaceVertexCData',obj.scalpData);
            
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
        end
        %%
        function plotStep(obj,step)
            nowCursor = obj.nowCursor + step;
            if nowCursor > obj.streamHandle.timeStamp(end)
                nowCursor = obj.streamHandle.timeStamp(end);
            end
            obj.plotThisTimeStamp(nowCursor);
        end
        function obj = changeSettings(obj), disp('Nothing to configure.');end
        %%
        function rePaint(obj,hObject,opt)
            CData = get(hObject,'userData');
            if isempty(strfind(opt,'Off'))
                set(hObject,'CData',CData{2});
            else
                set(hObject,'CData',CData{1});
            end
            switch opt
                case 'labelsOn'
                    set(obj.hLabels,'Visible','on');
                case 'labelsOff'
                    set(obj.hLabels,'Visible','off');
                case 'sensorsOn'
                    set(obj.hSensors,'Visible','on');
                case 'sensorsOff'
                    set(obj.hSensors,'Visible','off');
                case 'scalpOn'
                    val = obj.scalpData;
                    mx = 0.9*max(abs(val));
                    set(obj.hCortex,'FaceAlpha',0.15);
                    if strcmp(get(obj.hVector,'Visible'),'on')
                        set(obj.hScalp,'Visible','on','FaceAlpha',0.65);
                    else
                        set(obj.hScalp,'Visible','on','FaceAlpha',0.85);
                    end
                    if isempty(val), return;end
                    % set(get(obj.hScalp,'Parent'),'Clim',[-mx mx]);
                case 'scalpOff'
                    val = obj.sourceMagnitud;
                    mx = 0.9*max(abs(val));
                    set(obj.hScalp,'Visible','off');
                    set(obj.hCortex,'Visible','on','FaceAlpha',1);
                    % set(get(obj.hCortex,'Parent'),'Clim',[-mx mx]);
                case 'vectorOn'
                    set(obj.hVector,'Visible','on');
                    set(obj.hCortex,'FaceAlpha',0.75);
                case 'vectorOff'
                    set(obj.hVector,'Visible','off');
                    set(obj.hCortex,'FaceAlpha',1);
            end
        end
        %%
        function output_txt = showLabel(obj,event_obj)
            persistent DT
            if strcmp(obj.dcmHandle.Enable,'off'),return;end
            if strcmp(get(obj.hVector,'Visible'),'on')
                set(obj.hVector,'Visible','off');
                set(obj.hCortex,'FaceAlpha',1);
            end
            if isempty(DT)
                load(obj.streamHandle.surfaces);
                vertices = surfData(3).vertices;
                DT = DelaunayTri(vertices(:,1),vertices(:,2),vertices(:,3));
            end
            pos = get(event_obj,'Position');
            loc = nearestNeighbor(DT, pos);
            output_txt = obj.streamHandle.atlas.label{obj.streamHandle.atlas.color(loc)};
            drawnow
        end
    end
end