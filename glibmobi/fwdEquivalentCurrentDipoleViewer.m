% For details visit:  https://code.google.com/p/mobilab/
% 
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Jan-2012

classdef fwdEquivalentCurrentDipoleViewer < handle
    properties
        hFigure
        hAxes
        streamObj
        hLabels
        hSensors
        hScalp
        hCortex
        hVector
        hDipoles
        dcmHandle
        ecd
        xyz
        pointer
        interpolator
    end
    methods
        function obj = fwdEquivalentCurrentDipoleViewer(streamObj,xyz,ecd,figureTitle)
            if nargin < 2, error('Not enough input arguments.');end
            N = size(xyz,1);
            if nargin < 3, ecd = ones(N,3);end
            if nargin < 4, figureTitle = '';end
            if isa(streamObj,'pcdStream'), channelLabels = streamObj.parent.label;else channelLabels = streamObj.getChannelLabels;end
            
            obj.streamObj = streamObj; 
            load(obj.streamObj.surfaces);
            color = [0.93 0.96 1];
            
            mPath = which('mobilabApplication');
            if isempty(mPath)
                 path = fileparts(which('equivalentCurrentDipoleViewer'));
            else path = fullfile(fileparts(mPath),'skin');
            end
            
            try labelsOn  = imread([path filesep 'labelsOn.png']);
                labelsOff = imread([path filesep 'labelsOff.png']);
                sensorsOn = imread([path filesep 'sensorsOn.png']);
                sensorsOff = imread([path filesep 'sensorsOff.png']);
                scalpOn = imread([path filesep 'scalpOn.png']);
                scalpOff = imread([path filesep 'scalpOff.png']);
                prev = imread([path filesep '32px-Gnome-media-seek-backward.svg.png']);
                next = imread([path filesep '32px-Gnome-media-seek-forward.svg.png']);
            catch ME
                ME.message = sprintf('%s\nSome icons may be missing.',ME.message);
                ME.rethrow;
            end
            if isa(streamObj,'struct'), visible = 'off';else visible = 'on';end
            obj.hFigure = figure('Menubar','figure','ToolBar','figure','renderer','opengl','Visible',visible,'Color',color,'Name',figureTitle);
            position = get(obj.hFigure,'Position');
            set(obj.hFigure,'Position',[position(1:2) 1.06*position(3:4)]);
            obj.hAxes = axes('Parent',obj.hFigure);         
            toolbarHandle = findall(obj.hFigure,'Type','uitoolbar');
            
            hcb(1) = uitoggletool(toolbarHandle,'CData',labelsOff,'Separator','on','HandleVisibility','off','TooltipString','Labels On/Off','userData',{labelsOn,labelsOff},'State','off');
            set(hcb(1),'OnCallback',@(src,event)rePaint(obj,hcb(1),'labelsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(1),'labelsOff'));
            
            hcb(2) = uitoggletool(toolbarHandle,'CData',sensorsOff,'Separator','off','HandleVisibility','off','TooltipString','Sensors On/Off','userData',{sensorsOn,sensorsOff},'State','off');
            set(hcb(2),'OnCallback',@(src,event)rePaint(obj,hcb(2),'sensorsOn'),'OffCallback',@(src, event)rePaint(obj,hcb(2),'sensorsOff'));
            
            hcb(3) = uitoggletool(toolbarHandle,'CData',scalpOff,'Separator','off','HandleVisibility','off','TooltipString','Scalp On/Off','userData',{scalpOn,scalpOff},'State','off');
            set(hcb(3),'OnCallback',@(src,event)rePaint(obj,hcb(3),'scalpOn'),'OffCallback',@(src, event)rePaint(obj,hcb(3),'scalpOff'));
            
            uipushtool(toolbarHandle,'CData',prev,'Separator','off','HandleVisibility','off','TooltipString','Previous','ClickedCallback',@obj.prev);
            uipushtool(toolbarHandle,'CData',next,'Separator','off','HandleVisibility','off','TooltipString','Next','ClickedCallback',@obj.next);
            set(obj.hFigure,'WindowScrollWheelFcn',@(src, event)mouseMove(obj,[], event));      
            
            obj.dcmHandle = datacursormode(obj.hFigure);
            obj.dcmHandle.SnapToDataVertex = 'off';
            set(obj.dcmHandle,'UpdateFcn',@(src,event)showLabel(obj,event));
            obj.dcmHandle.Enable = 'off';
            hold(obj.hAxes,'on');
            
            obj.hSensors = scatter3(obj.hAxes,obj.streamObj.channelSpace(:,1),obj.streamObj.channelSpace(:,2),...
                obj.streamObj.channelSpace(:,3),'filled','MarkerEdgeColor','k','MarkerFaceColor','y');
            set(obj.hSensors,'Visible','off');
            
            N = length(channelLabels);
            k = 1.1;
            obj.hLabels = zeros(N,1);
            for it=1:N, obj.hLabels(it) = text('Position',k*obj.streamObj.channelSpace(it,:),'String',channelLabels{it},'Parent',obj.hAxes);end
            set(obj.hLabels,'Visible','off');
            
            obj.ecd = ecd;
            obj.xyz = xyz;
            skinColor = [1 0.75 0.65];
            obj.pointer = 1;
            obj.interpolator = geometricTools.localGaussianInterpolator(obj.streamObj.channelSpace,surfData(1).vertices,6);
            
            % dipoles
            dp = std(squeeze(ecd(:,1,:)));
            dp = dp/max(dp)*5;
            [sx,sy,sz] = ellipsoid(xyz(1,1),xyz(1,2),xyz(1,3),dp(1),dp(2),dp(3));
            obj.hDipoles = surf(obj.hAxes,sx,sy,sz,'LineStyle','none','FaceColor','y');
            
            % vectors
            hvx = quiver3(xyz(1,1),xyz(1,2),xyz(1,3),dp(1),0*dp(2),0*dp(3),0.25,'r','LineWidth',2);
            hvy = quiver3(xyz(1,1),xyz(1,2),xyz(1,3),0*dp(1),dp(2),0*dp(3),0.25,'g','LineWidth',2);
            hvz = quiver3(xyz(1,1),xyz(1,2),xyz(1,3),0*dp(:,1),0*dp(:,2),dp(3),0.25,'b','LineWidth',2);
            obj.hVector = [hvx;hvy;hvz];
            
            % cortex
            obj.hCortex = patch('vertices',surfData(3).vertices,'faces',surfData(3).faces,'FaceColor',skinColor,...
                'FaceLighting','phong','LineStyle','none','FaceAlpha',0.5,'SpecularColorReflectance',0,...
                'SpecularExponent',50,'SpecularStrength',0.5,'Parent',obj.hAxes);
            camlight(0,180)
            camlight(0,0)
                        
            % scalp
            val = sum( (obj.interpolator*squeeze(obj.ecd(:,1,:))).^2,2);
            obj.hScalp = patch('vertices',surfData(1).vertices,'faces',surfData(1).faces,'FaceVertexCData',val,...
                'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',0.75,...
                'SpecularColorReflectance',0,'SpecularExponent',50,'SpecularStrength',0.5,'Parent',obj.hAxes,'Visible','on');
            
            mx = max(abs(val));
            set(obj.hAxes,'Clim',[-mx mx]);
            view(obj.hAxes,[90 0]);
            hold(obj.hAxes,'off');
            axis(obj.hAxes,'equal','vis3d');
            axis(obj.hAxes,'off')
            if isprop(obj.streamObj,'name'), objName = [streamObj.name ': '];else objName = '';end
            set(obj.hFigure,'Visible',visible,'userData',obj,'Name',['ECD ' objName]);
            rotate3d
            drawnow
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
                case 'labelsOn',   set(obj.hLabels,'Visible','on');
                case 'labelsOff',  set(obj.hLabels,'Visible','off');
                case 'sensorsOn',  set(obj.hSensors,'Visible','on');
                case 'sensorsOff', set(obj.hSensors,'Visible','off');
                case 'scalpOn',    set(obj.hScalp,'Visible','on');
                case 'scalpOff',   set(obj.hScalp,'Visible','off');
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
            output_txt = obj.streamObj.atlas.label{obj.streamObj.atlas.colorTable(loc)};
            drawnow
        end
        %
        function prev(obj,~,~)
            obj.pointer = obj.pointer-1;
            if obj.pointer < 1, obj.pointer = 1;end
            val = squeeze(obj.ecd(:,obj.pointer,:));
            val = sum( (obj.interpolator*val).^2,2);
            set(obj.hScalp,'FaceVertexCData',val);
            mx = max(abs(val));
            set(obj.hAxes,'Clim',[-mx mx]);
            
            % dipoles
            dp = std(squeeze(obj.ecd(:,obj.pointer,:)));
            dp = dp/max(dp);
            [sx,sy,sz] = ellipsoid(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),dp(2),dp(3));
            set(obj.hDipoles,'XData',sx,'YData',sy,'ZData',sz);
            
            % vectors
            hold(obj.hAxes,'on');
            try delete(obj.hVector);end %#ok
            hvx = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),0*dp(2),0*dp(3),0.25,'r','LineWidth',2);
            hvy = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),dp(2),0*dp(3),0.25,'g','LineWidth',2);
            hvz = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),0*dp(2),dp(3),0.25,'b','LineWidth',2);
            obj.hVector = [hvx;hvy;hvz];
            hold(obj.hAxes,'off');
            if isprop(obj.streamObj,'name'), objName = [obj.streamObj.name ': '];else objName = '';end
            set(obj.hFigure,'Name',[objName num2str(obj.pointer) '/' num2str(size(obj.xyz,1))]);
        end
        %%
        function next(obj,~,~)
            obj.pointer = obj.pointer+1;
            n = size(obj.xyz,1);
            if obj.pointer > n, obj.pointer = n;end
            val = squeeze(obj.ecd(:,obj.pointer,:));
            val = sum( (obj.interpolator*val).^2,2);
            set(obj.hScalp,'FaceVertexCData',val);
            mx = max(abs(val));
            set(obj.hAxes,'Clim',[-mx mx]);
            
            % dipoles
            dp = std(squeeze(obj.ecd(:,obj.pointer,:)));
            dp = dp/max(dp);
            [sx,sy,sz] = ellipsoid(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),dp(2),dp(3));
            set(obj.hDipoles,'XData',sx,'YData',sy,'ZData',sz);
            
            % vectors
            hold(obj.hAxes,'on');
            try delete(obj.hVector);end %#ok
            hvx = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),0*dp(2),0*dp(3),0.25,'r','LineWidth',2);
            hvy = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),dp(2),0*dp(3),0.25,'g','LineWidth',2);
            hvz = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),0*dp(2),dp(3),0.25,'b','LineWidth',2);
            obj.hVector = [hvx;hvy;hvz];
            hold(obj.hAxes,'off');
            if isprop(obj.streamObj,'name'), objName = [obj.streamObj.name ': '];else objName = '';end
            set(obj.hFigure,'Name',[objName num2str(obj.pointer) '/' num2str(size(obj.xyz,1))]);
        end
        %%
        function mouseMove(obj,~,eventObj)
            obj.pointer = obj.pointer - eventObj.VerticalScrollCount;%*eventObj.VerticalScrollAmount;
            if obj.pointer < 1, obj.pointer = 1;end
            if obj.pointer > size(obj.xyz,1), obj.pointer = size(obj.xyz,1);end
            
            val = squeeze(obj.ecd(:,obj.pointer,:));
            val = sum( (obj.interpolator*val).^2,2);
            set(obj.hScalp,'FaceVertexCData',val);
            mx = max(abs(val));
            set(obj.hAxes,'Clim',[-mx mx]);
            
            % dipoles
            dp = std(squeeze(obj.ecd(:,obj.pointer,:)));
            dp = dp/max(dp);
            [sx,sy,sz] = ellipsoid(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),dp(2),dp(3));
            set(obj.hDipoles,'XData',sx,'YData',sy,'ZData',sz);
            
            % vectors
            hold(obj.hAxes,'on');
            try delete(obj.hVector);end %#ok
            hvx = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),dp(1),0*dp(2),0*dp(3),0.25,'r','LineWidth',2);
            hvy = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),dp(2),0*dp(3),0.25,'g','LineWidth',2);
            hvz = quiver3(obj.xyz(obj.pointer,1),obj.xyz(obj.pointer,2),obj.xyz(obj.pointer,3),0*dp(1),0*dp(2),dp(3),0.25,'b','LineWidth',2);
            obj.hVector = [hvx;hvy;hvz];
            hold(obj.hAxes,'off');
            if isprop(obj.streamObj,'name'), objName = [obj.streamObj.name ': '];else objName = '';end
            set(obj.hFigure,'Name',[objName num2str(obj.pointer) '/' num2str(size(obj.xyz,1))]);
        end
    end
end
