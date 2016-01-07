classdef topographyBrowserHandle < browserHandle
    properties
        hScalp
        hSensors
        hLabels
        numberOfChannelsToPlot
        cmap = 'bipolar';
        showLabels = true;
        interpolator
        surfData
        faceAlpha = 1;
        showSensors = true;
    end
    properties(SetObservable)
        channelIndex
    end
    methods
        %% constructor
        function obj = topographyBrowserHandle(dStreamObj,defaults)
            if nargin < 2,
                defaults.startTime = dStreamObj.timeStamp(1);
                defaults.endTime = dStreamObj.timeStamp(end);
                defaults.step = 1;
                defaults.channels = 1:dStreamObj.numberOfChannels;
                defaults.mode = 'standalone'; 
                defaults.speed = 1;
                defaults.font = struct('size',12,'weight','normal');
                defaults.cmap = 'jet';
            end
            if ~isfield(defaults,'uuid'), defaults.uuid = dStreamObj.uuid;end
            if ~isfield(defaults,'startTime'), defaults.startTime = dStreamObj.timeStamp(1);end
            if ~isfield(defaults,'endTime'), defaults.endTime = dStreamObj.timeStamp(end);end
            if ~isfield(defaults,'step'), defaults.step = 1;end
            if ~isfield(defaults,'channels'), defaults.channels = 1:dStreamObj.numberOfChannels;end
            if ~isfield(defaults,'mode'), defaults.mode = 'standalone';end
            if ~isfield(defaults,'speed'), defaults.speed = 1;end
            if ~isfield(defaults,'font'), defaults.font = struct('size',12,'weight','normal');end
            if ~isfield(defaults,'cmap'), defaults.cmap = 'bipolar';end
            %obj.timerMult = 0.1;
            obj.uuid = defaults.uuid; 
            obj.streamHandle = dStreamObj;
            obj.font = defaults.font;
            
            obj.addlistener('timeIndex','PostSet',@topographyBrowserHandle.updateTimeIndexDependencies);
            obj.addlistener('font','PostSet',@topographyBrowserHandle.updateFont);
            
            obj.speed = defaults.speed;
            obj.state = false;
                                    
            [t1,t2] = dStreamObj.getTimeIndex([defaults.startTime defaults.endTime]);
            obj.timeIndex = t1:t2; 
            obj.step = 1;       % one second
            obj.nowCursor = dStreamObj.timeStamp(obj.timeIndex(1)) + 2.5;
                        
            obj.channelIndex = 1:dStreamObj.numberOfChannels; 
            obj.cmap = defaults.cmap;
            if strcmp(defaults.mode,'standalone'), obj.master = -1;end
            
            load(obj.streamHandle.surfaces);
            obj.surfData = surfData(1); %#ok
            obj.figureHandle = streamBrowserNG(obj);
        end
        %%
        function defaults = saveobj(obj)
            defaults = saveobj@browserHandle(obj);
            defaults.cmap = obj.cmap;
            defaults.showLabels = obj.showLabels;
            defaults.streamName = obj.streamHandle.name;
            defaults.browserType = 'topographyBrowser';
            defaults.channels = obj.channelIndex;
        end
        %% plot
        function t0 = createGraphicObjects(obj,nowCursor)
     
            createGraphicObjects@browserHandle(obj);
            set(obj.figureHandle,'RendererMode','manual')
            set(obj.figureHandle,'Renderer','opengl')
            box(obj.axesHandle,'off');
            
            load(obj.streamHandle.surfaces);
            obj.surfData = surfData(1); %#ok
            obj.interpolator = geometricTools.localGaussianInterpolator(obj.streamHandle.channelSpace,obj.surfData.vertices,32);
            if isa(obj.streamHandle,'icaStream'), obj.interpolator = obj.interpolator*obj.streamHandle.icawinv(:,obj.channelIndex);end
            
            if strcmp(obj.cmap,'bipolar'), colormap(bipolar(512, 0.99)); else colormap(obj.cmap);end
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex), obj.nowCursor);
            data = obj.streamHandle.data(obj.timeIndex(t0),obj.channelIndex)';
            data = double(data);            
            cla(obj.axesHandle);
            
            hold(obj.axesHandle,'on');
            obj.hScalp = patch('vertices',obj.surfData.vertices,'faces',obj.surfData.faces,'FaceVertexCData',obj.interpolator*data,...
                'FaceColor','interp','FaceLighting','phong','LineStyle','none','FaceAlpha',obj.faceAlpha,'SpecularColorReflectance',0,...
                    'SpecularExponent',50,'SpecularStrength',0.5,'Parent',obj.axesHandle);
            camlight(0,180)
            camlight(0,0)
            if obj.showSensors
                obj.hSensors = scatter3(obj.streamHandle.channelSpace(obj.channelIndex,1),obj.streamHandle.channelSpace(obj.channelIndex,2),...
                    obj.streamHandle.channelSpace(obj.channelIndex,3),'filled','MarkerFaceColor','w','MarkerEdgeColor','k','Parent',obj.axesHandle);
            else
                try delete(obj.hSensors);end%#ok
            end
            if obj.showLabels
                N = length(obj.streamHandle.label);
                k = 1.1;
                obj.hLabels = zeros(N,1);
                for it=1:N, obj.hLabels(it) = text('Position',k*obj.streamHandle.channelSpace(it,:),'String',obj.streamHandle.label{it},'Parent',obj.axesHandle);end
            else
                try delete(obj.hLabels);end%#ok
            end
            hold(obj.axesHandle,'off');
            title(obj.axesHandle,'');
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec'],'FontSize',obj.font.size,'FontWeight',obj.font.weight);
            axis(obj.axesHandle,'equal','vis3d');
            axis(obj.axesHandle,'off')
            rotate3d
        end
        %%
        function t0 = plotThisTimeStamp(obj,nowCursor)
            
            if nowCursor > obj.streamHandle.timeStamp(obj.timeIndex(end))
                nowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
                if strcmp(get(obj.timerObj,'Running'),'on')
                    stop(obj.timerObj);
                end
            end
            
            % find now cursor index
            obj.nowCursor = nowCursor;
            %[~,t0] = min(abs(obj.streamHandle.timeStamp(obj.timeIndex) - obj.nowCursor));
            t0 = binary_findClosest(obj.streamHandle.timeStamp(obj.timeIndex), obj.nowCursor);
            data = obj.streamHandle.data(obj.timeIndex(t0),obj.channelIndex)';
            data = double(data);
            val = obj.interpolator*data;
            mx = max(abs(val))+eps;
            set(obj.hScalp,'FaceVertexCData',val);
            set(obj.axesHandle,'Clim',[-mx mx]);
            set(obj.timeTexttHandle,'String',['Current latency = ' num2str(obj.nowCursor,4) ' sec']);
            set(obj.sliderHandle,'Value',obj.nowCursor);
        end
        %%
        function plotStep(obj,step)
            if obj.nowCursor+step < obj.streamHandle.timeStamp(obj.timeIndex(end)) &&...
                    obj.nowCursor+step > obj.streamHandle.timeStamp(obj.timeIndex(1))
                newNowCursor = obj.nowCursor+step;
            elseif obj.nowCursor+step > obj.streamHandle.timeStamp(obj.timeIndex(end))
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(end));
            else
                newNowCursor = obj.streamHandle.timeStamp(obj.timeIndex(1));
            end
            obj.plotThisTimeStamp(newNowCursor);
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
        function obj = changeSettings(obj) 
            sg = sign(double(obj.speed<1));
            sg(sg==0) = -1;
            speed1 = [num2str(-sg*obj.speed^(-sg)) 'x'];
            speed2 = [1/5 1/4 1/3 1/2 1 2 3 4 5];
            
            prefObj = [...
                PropertyGridField('channels',obj.channelIndex,'DisplayName','Channels to plot','Description','This field accepts matlab code returning a subset of channels, for instance use: ''setdiff(1:10,[3 5 7])'' to plot channels from 1 to 10 excepting 3, 5, and 7.')...
                PropertyGridField('speed',speed1,'Type',PropertyType('char','row',{'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'}),'DisplayName','Speed','Description','Speed of the play mode.')...
                PropertyGridField('faceAlpha',obj.faceAlpha,'DisplayName','Face alpha','Description','(Patch transparency.')...
                PropertyGridField('showSensors',obj.showSensors,'DisplayName','Show sensors','Description','(Shows the sensor positions.')...
                PropertyGridField('showLabels',obj.showLabels,'DisplayName','Show sensor labels','Description','(Shows the sensor labels.')...
                PropertyGridField('colormap',obj.cmap,'Type',PropertyType('char','row',{'bipolar','jet','hsv','hot','cool','spring','summer'}),'DisplayName','Colormap','Description','')...
                ];
         
            % create figure
            f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
            position = get(f,'position');
            set(f,'position',[position(1:2) 385 424]);
            g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
            uiwait(f); % wait for figure to close
            val = g.GetPropertyValues();
         
            obj.speed = speed2(ismember({'-5x','-4x','-3x','-2x','1x','2x','3x','4x','5x'},val.speed));
            obj.faceAlpha = val.faceAlpha;
            obj.cmap = val.colormap;
            obj.channelIndex = val.channels;
            obj.showSensors = val.showSensors;
            obj.showLabels = val.showLabels;
            
            figure(obj.figureHandle);
            obj.createGraphicObjects(obj.nowCursor);
        end
        %%
        function set.channelIndex(obj,channelIndex)
            I = ismember(1:obj.streamHandle.numberOfChannels,channelIndex);
            if ~any(I), return;end
            obj.channelIndex = channelIndex;
        end
    end
    %%
    methods(Static)
        %%
        function updateTimeIndexDependencies(~,evnt)
            if evnt.AffectedObject.timeIndex(1) ~= -1
                evnt.AffectedObject.nowCursor = evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)) + 2.5;
                set(evnt.AffectedObject.sliderHandle,'Min',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)));
                set(evnt.AffectedObject.sliderHandle,'Max',evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)));
                set(evnt.AffectedObject.sliderHandle,'Value',evnt.AffectedObject.nowCursor);
            end
        end
        %%
        function updateFont(~,evnt)
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text4'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(1)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(findobj(evnt.AffectedObject.figureHandle,'tag','text5'),'String',num2str(evnt.AffectedObject.streamHandle.timeStamp(evnt.AffectedObject.timeIndex(end)),4),...
                'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            
            set(evnt.AffectedObject.timeTexttHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            set(evnt.AffectedObject.axesHandle,'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight)
            
            xlabel(evnt.AffectedObject.axesHandle,'x','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            zlabel(evnt.AffectedObject.axesHandle,'y','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
            ylabel(evnt.AffectedObject.axesHandle,'z','Color',[0 0 0.4],'FontSize',evnt.AffectedObject.font.size,'FontWeight',evnt.AffectedObject.font.weight);
        end
    end
end
