classdef courrentSourceViewer < handle
    properties
        hFigure
        hAxes
        streamObj
        hLabels
        hSensors
        hScalp
        hCortex
        hVector
        dcmHandle
    end
    methods
        function obj = courrentSourceViewer(streamObj,J,V,figureTitle)
            if nargin < 3, V = [];end
            if nargin < 4, figureTitle = '';end
            obj.streamObj = streamObj; 
            color = [0.93 0.96 1];
            if isa(streamObj,'eeg') || isa(streamObj,'struct')
                mPath = which('mobilabApplication');
                path = fullfile(fileparts(mPath),'skin');
                labelsOn  = imread([path filesep 'labelsOn.png']);
                labelsOff = imread([path filesep 'labelsOff.png']);
                sensorsOn = imread([path filesep 'sensorsOn.png']);
                sensorsOff = imread([path filesep 'sensorsOff.png']);
                scalpOn = imread([path filesep 'scalpOn.png']);
                scalpOff = imread([path filesep 'scalpOff.png']);
                vectorOn = imread([path filesep 'vectorOn.png']);
                vectorOff = imread([path filesep 'vectorOff.png']);
            else
                labelsOn  = rand(22);
                labelsOff = rand(22);
                sensorsOn = rand(22);
                sensorsOff = rand(22);
                scalpOn = rand(22);
                scalpOff = rand(22);
                vectorOn = rand(22);
                vectorOff = rand(22);
            end
            if isa(streamObj,'struct'), visible = 'off';else visible = 'on';end
            obj.hFigure = figure('Menubar','figure','ToolBar','figure','renderer','opengl','Visible',visible,'Color',color);
            obj.hAxes = axes('Parent',obj.hFigure);
                       
            toolbarHandle = findall(obj.hFigure,'Type','uitoolbar');
            
            hcb(1) = uitoggletool(toolbarHandle,'CData',labelsOff,'Separator','on','HandleVisibility','off','TooltipString','Labels On/Off','userData',{labelsOn,labelsOff},'State','off');
            set(hcb(1),'OnCallback',@(src,event)rePaint(obj,hcb(1),'labelsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(1),'labelsOff'));
            
            hcb(2) = uitoggletool(toolbarHandle,'CData',sensorsOff,'Separator','off','HandleVisibility','off','TooltipString','Sensors On/Off','userData',{sensorsOn,sensorsOff},'State','off');
            set(hcb(2),'OnCallback',@(src,event)rePaint(obj,hcb(2),'sensorsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(2),'sensorsOff'));
            
            hcb(3) = uitoggletool(toolbarHandle,'CData',scalpOff,'Separator','off','HandleVisibility','off','TooltipString','Scalp On/Off','userData',{scalpOn,scalpOff},'State','off');
            set(hcb(3),'OnCallback',@(src,event)rePaint(obj,hcb(3),'scalpOn'),'OffCallback',@(src, event)rePaint(obj,hcb(3),'scalpOff'));
            
            hcb(4) = uitoggletool(toolbarHandle,'CData',vectorOff,'Separator','off','HandleVisibility','off','TooltipString','Scalp On/Off','userData',{vectorOn,vectorOff},'State','off');
            set(hcb(4),'OnCallback',@(src,event)rePaint(obj,hcb(4),'vectorOn'),'OffCallback',@(src, event)rePaint(obj,hcb(4),'vectorOff'));
            
            obj.dcmHandle = datacursormode(obj.hFigure);
            obj.dcmHandle.SnapToDataVertex = 'off';
            set(obj.dcmHandle,'UpdateFcn',@(src,event)showLabel(obj,event));
            obj.dcmHandle.Enable = 'off';
            
            hold(obj.hAxes,'on');
            
            obj.hSensors = scatter3(obj.hAxes,obj.streamObj.channelSpace(:,1),obj.streamObj.channelSpace(:,2),...
                obj.streamObj.channelSpace(:,3),'filled','MarkerEdgeColor','k','MarkerFaceColor','y');
            set(obj.hSensors,'Visible','off');
            
            N = length(obj.streamObj.label);
            k = 1.1;
            obj.hLabels = zeros(N,1);
            for it=1:N, obj.hLabels(it) = text('Position',k*obj.streamObj.channelSpace(it,:),'String',obj.streamObj.label{it},'Parent',obj.hAxes);end
            set(obj.hLabels,'Visible','off');
            
            load(obj.streamObj.surfaces);
                        
            % vectors
            normalsIn = false;
            normals = geometricTools.getSurfaceNormals(surfData(3).vertices,surfData(3).faces,normalsIn);
            if size(J,1) == 3*size(surfData(3).vertices,1)  
                J = reshape(J,[size(J,1)/3 3]);
                Jm = sqrt(sum(J.^2,2));
                s = sign(dot(normals,J,2));
                Jm = s.*Jm;
                obj.hVector = quiver3(surfData(3).vertices(:,1),surfData(3).vertices(:,2),surfData(3).vertices(:,3),J(:,1),J(:,2),J(:,3),2);
            else
                Jm = J;
                obj.hVector = quiver3(surfData(3).vertices(:,1),surfData(3).vertices(:,2),surfData(3).vertices(:,3),normals(:,1),normals(:,2),normals(:,3),2);
            end
            set(obj.hVector,'Color','k','Visible','off');
            
            % cortex
            obj.hCortex = patch('vertices',surfData(3).vertices,'faces',surfData(3).faces,'FaceVertexCData',Jm,...
                'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',1,'Parent',obj.hAxes);
            camlight(0,180)
            camlight(0,0)
            mx = max(Jm(:));
            % scalp
            if isempty(V)
                skinColor = [1,.75,.65];
                obj.hScalp = patch('vertices',surfData(1).vertices,'faces',surfData(1).faces,'facecolor',skinColor,...
                    'facelighting','phong','LineStyle','none','FaceAlpha',.85,'Parent',obj.hAxes,'Visible','off');
            else
                %Vi = geometricTools.splineInterpolator(obj.streamObj.channelSpace,V,surfData(1).vertices);
                W = geometricTools.localGaussianInterpolator(obj.streamObj.channelSpace,surfData(1).vertices,6);
                Vi = W*V;
                obj.hScalp = patch('vertices',surfData(1).vertices,'faces',surfData(1).faces,'FaceVertexCData',Vi,...
                    'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',0.85,'Parent',obj.hAxes,'Visible','off');
            end
            view(obj.hAxes,[90 0]);
            colorbar
            % box on;
            title(texlabel(figureTitle,'literal'));
            hold(obj.hAxes,'off');
            axis(obj.hAxes,'equal');
            axis(obj.hAxes,'off')
            set(obj.hAxes,'Clim',[-mx mx]);
            set(obj.hFigure,'Visible',visible,'userData',obj);
            rotate3d
        end
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
                    val = get(obj.hScalp,'FaceVertexCData');
                    mx = max(abs([min(val) max(val)]));
                    %set(obj.hCortex,'Visible','off');
                    set(obj.hScalp,'Visible','on');
                    set(get(obj.hScalp,'Parent'),'Clim',[-mx mx]);
                case 'scalpOff'
                    val = get(obj.hCortex,'FaceVertexCData');
                    mx = max(val(:));
                    set(obj.hScalp,'Visible','off');
                    set(obj.hCortex,'Visible','on');
                    set(get(obj.hCortex,'Parent'),'Clim',[-mx mx]);
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
            if isempty(DT)
                load(obj.streamObj.surfaces);
                vertices = surfData(3).vertices;
                DT = DelaunayTri(vertices(:,1),vertices(:,2),vertices(:,3));
            end
            pos = get(event_obj,'Position');
            loc = nearestNeighbor(DT, pos);
            output_txt = obj.streamObj.atlas.label{obj.streamObj.atlas.color(loc)};
            %updateCursor(obj.dcmHandle,pos);
        end
    end
end