%% mocap class
% Creates and uses a mocap object.
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 05-Apr-2011

%%
classdef mocap < dataStream
    properties(GetAccess = public, SetAccess = public)
        animationParameters; % Structure used for animating stick figures. 
                             % It should have the following fields:
                             % limits: [min _x max_x; min_y max_y; min_z max_z] 
                             %         specifying the dimensions of the mocap space
                             % conn:   [marker_i marker_j] matrix specifying connections
                             %         between markers.
                             
        lsMarker2JointMapping
    end
    properties(GetAccess = public, SetAccess = protected, AbortSet = true)
        bodyModelFile
    end
    properties(Dependent)
        dataInXYZ           % Dependent property that reshapes the second dimension
                            % of the field data to allows accessing directly xyz 
                            % coordinates of motion capture markers.
                            
        magnitude           % Dependent property that computes the magnitude (distance
                            % from the origin) of xyz motion capture markers.
    end
    methods
        %%
        function obj = mocap(header)
            % Creates a mocap object.
            % 
            % Input arguments:
            %       header: header file (string)
            % 
            % Output arguments:
            %       obj: mocap object (handle)

            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        %%
        function animationParameters = get.animationParameters(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.animationParameters'))
                animationParameters = obj.animationParameters;
                return;
            end
            if isempty(obj.animationParameters)
                try obj.animationParameters = retrieveProperty(obj,'animationParameters');
                catch
                    animationParameters = struct('limits',[],'conn',[],'bodymodel',[]);
                    save(obj.header,'-mat','-append','animationParameters')
                    obj.animationParameters = animationParameters;
                end
            end
            saveIt = false;
            if ~isfield(obj.animationParameters,'limits'), obj.animationParameters.limits = [];saveIt=true;end
            if ~isfield(obj.animationParameters,'conn'), obj.animationParameters.conn = [];saveIt=true;end
            if ~isfield(obj.animationParameters,'bodymodel'), obj.animationParameters.bodymodel = [];saveIt=true;end
            if isempty(obj.animationParameters.limits)
                mx  = 1.05*max(max(abs(squeeze(obj.dataInXYZ(1:100:end,1,:)))));
                my  = 1.05*max(max(abs(squeeze(obj.dataInXYZ(1:100:end,2,:)))));
                mz  = 1.05*max(max(abs(squeeze(obj.dataInXYZ(1:100:end,3,:)))));
                mnz = min(min(squeeze(obj.dataInXYZ(1:100:end,3,:))));
                
                obj.animationParameters.limits = [-mx mx;-my my;mnz mz];
            end
            animationParameters = obj.animationParameters;
            if saveIt, save(obj.header,'-mat','-append','animationParameters');end
        end
        function  set.animationParameters(obj,animationParameters)
            stack = dbstack;            
            if any(strcmp({stack.name},'coreStreamObject.get.animationParameters'))
                obj.animationParameters = animationParametersj;
                return;
            end
            if ~isfield(animationParameters,'limits'), animationParameters.limits = [];end
            if ~isfield(animationParameters,'conn'), animationParameters.conn = [];end
            if ~isfield(animationParameters,'bodymodel'), animationParameters.bodymodel = [];end
            saveProperty(obj,'animationParameters',animationParameters);
            obj.animationParameters = animationParameters;
        end
        %%
        function lsMarker2JointMapping = get.lsMarker2JointMapping(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.set.lsMarker2JointMapping'))
                lsMarker2JointMapping = obj.lsMarker2JointMapping;
                return;
            end
            if isempty(obj.lsMarker2JointMapping), obj.lsMarker2JointMapping = retrieveProperty(obj,'lsMarker2JointMapping');end
            lsMarker2JointMapping = obj.lsMarker2JointMapping;
        end
        function set.lsMarker2JointMapping(obj,lsMarker2JointMapping)
            stack = dbstack;
            if any(strcmp({stack.name},'coreStreamObject.get.lsMarker2JointMapping'))
                obj.lsMarker2JointMapping = lsMarker2JointMapping;
                return;
            end
            save(obj.header,'-mat','-append','lsMarker2JointMapping');
            obj.lsMarker2JointMapping = lsMarker2JointMapping;
        end
        %%
        function data = get.dataInXYZ(obj)
            data = [];
            if obj.isMemoryMappingActive
                if obj.numberOfChannels/3 > 1
                    perm = 1:3;
                    if isempty(obj.hardwareMetaData.name) || strcmpi(obj.hardwareMetaData.name,'phasespace') %&& isa(obj.hardwareMetaData,'hardwareMetaData')
                        perm = [1 3 2];
                    elseif isempty(obj.hardwareMetaData.name) ||  ~isempty(strfind(obj.hardwareMetaData.name,'KinectMocap')) 
                        perm = [1 3 2];
                    end
                    %if strcmp(obj.hardwareMetaData.name,'optitrack')
                    %    perm = 1:3;%perm = [3 1 2];
                    %else
                    %    perm = [1 3 2];
                    %end
                    dim = obj.size;
                    obj.reshape([dim(1) 3 dim(2)/3]);
                    data = obj.mmfObj.Data.x(:,perm,:);
                    obj.reshape(dim);
                else data = obj.mmfObj.Data.x;
                end
            else disp('Cannot read the binary file.');
            end
        end
        %%
        function set.dataInXYZ(obj,data)
            obj.mmfObj.Writable = obj.writable;
            if strcmpi(obj.hardwareMetaData.name,'phasespace') && isa(obj.hardwareMetaData,'hardwareMetaData')
                perm = [1 3 2];
            else perm = [1 2 3];
            end
            if obj.numberOfChannels/3 > 1
                dim = obj.size;
                obj.reshape([dim(1) 3 dim(2)/3]);
                obj.mmfObj.Data.x = data(:,perm,:);
                obj.reshape(dim);
            else obj.mmfObj.Data.x = data(:,perm,:);
            end
            obj.mmfObj.Writable = false;
        end
        %%
        function mag = get.magnitude(obj)
            dim = obj.size;
            if ~mod(obj.numberOfChannels,3)
                obj.reshape([dim(1) 3 dim(2)/3]);
            elseif ~mod(obj.numberOfChannels,2)
                obj.reshape([dim(1) 2 dim(2)/2]);
            end
            mag = squeeze(sqrt(sum(obj.mmfObj.Data.x.^2,2)));
            obj.reshape(dim);
        end
        %%
        function jsonObj = serialize(obj)
            metadata = saveobj(obj);
            metadata.class = class(obj);
            metadata.size = size(obj);
            metadata.event = obj.event.uniqueLabel;
            metadata.writable = double(metadata.writable);
            metadata.history = obj.history;
            if isempty(metadata.animationParameters.conn)
                metadata.hasStickFigure = 'no';
            else
                metadata.hasStickFigure = 'yes';
            end
            metadata = rmfield(metadata,'animationParameters');
            metadata = rmfield(metadata,{'parentCommand' 'timeStamp','hardwareMetaData'});
            jsonObj = savejson('',metadata,'ForceRootName', false);
        end
        %%
        function loadConnectedBody(obj,file)
            % Loads from a .mat file a matrix called 'connectedBody' containing
            % the connections among markers. It fills the connections to 
            % animationParameters.conn.
            % 
            % Input argument:
            %       file: Pointer to the .mat file containing the connections
            %
            % Usage:
            %       file = mobilab.preferences.mocap.stickFigure;
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       mocapObj.loadConnectedBody( file );
            %       plot( mocapObj );

            if nargin < 2
                [filename,path] = uigetfile2('*.mat','Select the file containing the connections',obj.container.container.preferences.mocap.stickFigure);
                if isnumeric(filename), return;end
                file = fullfile(path,filename);
            end
            if ~ischar(file)
                [filename,path] = uigetfile2('*.mat','Select the file containing the connections',obj.container.container.preferences.mocap.stickFigure);
                if isnumeric(filename), return;end
                file = fullfile(path,filename);
            end
            if ~exist(file,'file'), error('The file does''t exist.');end
            warning off %#ok
            load(file,'connectedBody');
            warning on %#ok
            if ~exist('connectedBody','var'), error(' Variable ''connectedBody'' not found. ');end
            if ~all(ismember(unique(connectedBody(:))',1:obj.numberOfChannels/3)) %#ok
                error('MoBILAB:stickFigureDoesntMatch','The stick figure doesn''t match the channels in the mocap object.');
            end
            obj.animationParameters.conn = connectedBody;
            saveProperty(obj,'animationParameters',obj.animationParameters);
        end 
        %%
        function cobj = removeOcclusionArtifact(obj,varargin)
            % Fills-in occluded time points. In optical mocap systems like PhaseSpace
            % (http://www.phasespace.com/), sometimes markers go missing from more than
            % one camera causing the system to put zero or some other predetermined value
            % in that sample. This method identifies the samples that are exactly zero and
            % interpolates them.
            %
            % Input arguments:
            %       method: could be nearest, linear, spline, or pchip, default: pchip
            %       channels: channels to fix if needed, default: all
            % 
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       method   = 'pchip';
            %       newMocapObj = mocapObj.removeOcclusionArtifact( method);
            %       
            %       figure;plot(mocapObj.timeStamp, [mocapObj.data(:,1) newMocapObj.data(:,1)])
            %       xlabel('Time (sec)');legend({mocapObj.name newMocapObj.name});

            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            dispCommand = false;

            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter interpolation method: (''pchip'', ''spline'', ''linear'', ''nearest'')'};
                dlg_title = 'Input parameters';
                num_lines = 1;
                def = {obj.container.container.preferences.mocap.interpolation};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                dispCommand = true;
            end
            
            if nargin < 2, method = 'pchip';                 else method = varargin{1};end
            if nargin < 3, channels = 1:obj.numberOfChannels;else channels = varargin{2};end
            if ~ischar(method), error('prog:input','First argument must be a string that specify the interpolation method.');end
            switch lower(method)
                case 'nearest'
                case 'linear'
                case 'spline'
                case 'pchip'
                otherwise
                    error('prog:input','Unknown interpolation method. Go to interp1 help page to see the alternatives.');
            end
            if ~isnumeric(channels), error('Invalid input argument.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid input channels.');end
            
            try
                data = obj.mmfObj.Data.x; 
                commandHistory.commandName = 'removeOcclusionArtifact';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = method;
                commandHistory.varargin{2} = channels; 
                cobj = obj.copyobj(commandHistory);
                cobj.mmfObj.Data.x = obj.mmfObj.Data.x;
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                % fill occlusions interpolating the signal
                obj.initStatusbar(1,cobj.numberOfChannels,'Filling-in occluded time points...');
                for it=1:cobj.numberOfChannels
                    indi = data(:,channels(it))==0;
                    if any(indi) && sum(indi) < length(obj.timeStamp)-1
                        cobj.mmfObj.Data.x(indi,it) = interp1(obj.timeStamp(~indi),data(~indi,channels(it)),obj.timeStamp(indi),method);
                        if ~data(end,channels(it))
                            loc = find(~indi, 1, 'last' );
                            cobj.mmfObj.Data.x(loc+1:end,it) = data(loc,channels(it));
                        end
                        if ~data(1,channels(it))
                            loc = find(~indi, 1, 'first' );
                            cobj.mmfObj.Data.x(1:loc-1,it) = data(loc,channels(it));
                        end
                    end
                    obj.statusbar(it);
                end
                cobj.mmfObj.Writable = false;
            catch ME
                obj.statusbar(obj.numberOfChannels);
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function cobj = lowpass(obj, varargin)
            % Filters the motion capture data with a zero-lag lowpass FIR 
            % filter calling the method filter defined in dataStream.
            % 
            % Input arguments:
            %       cutOff:   lowpass cutoff frequency (in Hz)
            %       channels: channel to filter, default: all
            % 
            % Output argument:
            %       cobj:      handle to the new object
            %
            % Usage:
            %       mocapObj     = mobilab.allStreams.item{ mocapItem };
            %       cutOff       = 6;  % lowpass at 6 Hz
            %       mocapObjFilt = mocapObj.lowpass( cutOff );
            %       
            %       figure;plot(mocapObj.timeStamp, [mocapObj.data(:,1) mocapObjFilt.data(:,1)])
            %       xlabel('Time (sec)');legend({mocapObj.name mocapObjFilt.name});

            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter the cutoff frequency:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = str2double(varargin{1});
                if isnan(varargin{1}), return;end
                dispCommand = true;
            end
            
            % Cutoff Frequency
            if nargin < 2, fc = 6; else fc = varargin{1};end
            if nargin < 3, channels = 1:obj.numberOfChannels; else channels = varargin{2};end
            if nargin < 4
                N = 128;
                disp('Third argument must be the length of the filter (integer type). Using the default: 128.');
            elseif isnumeric(varargin{3})
                 N = varargin{3};
            else N = 128;
            end
            try cobj = obj.filter('lowpass',fc,channels,N);
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function cobj = timeDerivative(obj,varargin)
            % Computes the time derivatives of motion capture data. It smooths
            % the signals after each order of derivation to minimize cumulative
            % precision errors. As smoother it uses a zero-lag FIR lowpass filter.
            % Each new derivative is stored in a new object.
            % 
            % Input arguments:
            %       order:     maximum order of derivation, default: 3 (1 = velocity, 
            %                  2 = acceleration, 3 = jerk)
            %       cutOff:    lowpass filter cutoff, default: 18 Hz.
            %
            % Output argument:
            %       cobj:      handle to the object containing the latest order of
            %                  derivation
            %
            % Uses:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       order    = 3;
            %       mocapObj.timeDerivative( order );
            %       
            %       figure;
            %       subplot(411);plot(mocapObj.timeStamp,mocapObj.data(:,1));xlabel('Time (sec)');title(mocapObj.name)
            %       subplot(412);plot(mocapObj.timeStamp, mocapObj.children{1}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{1}.name)
            %       subplot(413);plot(mocapObj.timeStamp, mocapObj.children{2}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{2}.name)
            %       subplot(414);plot(mocapObj.timeStamp, mocapObj.children{3}.data(:,1));xlabel('Time (sec)');title(mocapObj.children{3}.name)

            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && isnumeric(varargin{1}) && length(varargin{1}) == 1 && varargin{1} == -1
                prompt = {'Enter the order of the time derivative: ( 1->vel, 2->acc, 3->jerk,... or 1:3->for all of them)'};
                dlg_title = 'Time derivative input parameter';
                num_lines = 1;
                def = {num2str(obj.container.container.preferences.mocap.derivationOrder)};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = eval(varargin{1});
                dispCommand = true;
            end
            if nargin < 2, order = 3;else order = varargin{1};end
            if nargin < 3, fc = 18;  else fc = varargin{2};end
            if nargin < 4, channels = 1:obj.numberOfChannels;else channels = varargin{3};end
            if nargin < 5, filterOrder = 128;else filterOrder = varargin{4};end
            if ~isnumeric(order), error('prog:input','First argument must be the order of the derivative (1=veloc, 2=acc, 3=jerk).');end
            if ~isnumeric(fc),    error('prog:input','Second argument must be the cut off frequency.');end
            if ~isnumeric(channels), error('Invalid channel.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid channel.');end
            Nch = length(channels);
            
            dt = 1/obj.samplingRate;
            dt = 1e3*dt; % from seconds to milliseconds
            order = unique(1:max(order));
            N = max(order);
            fc = min([fc obj.samplingRate/4]);
            try
                % smooth the data by 0 phase shifting moving average
                a = 1;
                b = obj.firDesign(filterOrder,'lowpass',fc);
                commandHistory.commandName = 'timeDerivative';
                commandHistory.uuid  = obj.uuid;
                commandHistory.varargin{2} = fc;
                commandHistory.varargin{3} = channels;
                obj.initStatusbar(1,N*Nch,'Computing time derivatives...');
                tmpObj = obj;
                for it=1:N
                    commandHistory.varargin{1} = it;
                    cobj = tmpObj.copyobj(commandHistory);
                    for jt=1:cobj.numberOfChannels
                        
                        % deriving
                        cobj.mmfObj.Data.x(1:end-1,jt) = diff(cobj.mmfObj.Data.x(:,jt),1)/dt;
                        cobj.mmfObj.Data.x(end,jt) = cobj.mmfObj.Data.x(end-1,jt);
                    
                        % smoothing
                        cobj.mmfObj.Data.x(:,jt) = filtfilt_fast(b,a,cobj.mmfObj.Data.x(:,jt));
                        obj.statusbar((Nch*(it-1)+jt));
                    end
                    tmpObj = cobj;
                end
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
            catch ME
                obj.statusbar(N*Nch);
                try obj.container.deleteItem(cobj.uuid);end %#ok
                ME.rethrow;
            end
        end
        %%
        function browserObj = plot(obj)
            % Overwrites the plot method defined in its base class to display
            % mocap data as stick figures in three dimensional space.
            
            browserObj = mocapBrowser(obj);
        end
        function browserObj = mocapBrowser(obj,defaults)
            if nargin < 2, defaults.browser  = @mocapBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser  = @mocapBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
        %%
        function cobj = projectDataPCA(obj,varargin)
            dispCommand = false;
            if  isnumeric(varargin{1}) && length(varargin{1}) == 1 && varargin{1} == -1
                
                prefObj = [...
                    PropertyGridField('startLatency',obj.timeStamp(1),'DisplayName','Start latency of clean segments','Description','Latency in seconds of the start of the clean segments of data.')...
                    PropertyGridField('endLatency',obj.timeStamp(end),'DisplayName','Final latency of clean segments','Description','Latency in seconds of the end of the clean segments of data.')...
                    PropertyGridField('robustPcaFlag',false,'DisplayName','Compute robust PCA','Description','Fits a Gaussian Mixture Model removing the component associated with the artefacts.')...
                    PropertyGridField('numberOfGaussianMixtures',3,'DisplayName','Number of gaussians','Description','Fits n+1 Gaussians. The last one is usually associated with the non-structured part of the data (noise).')...
                    ];
                
                % create figure
                hFigure = figure('MenuBar','none','Name','PCA','NumberTitle', 'off','Toolbar', 'none');
                position = get(hFigure,'position');
                set(hFigure,'position',[position(1:2) 303 231]);
                hPanel = uipanel(hFigure,'Title','','BackgroundColor','white','Units','pixels','Position',[0 55 303 180],'BorderType','none');
                g = PropertyGrid(hPanel,'Properties', prefObj,'Position', [0 0 1 1]);
                uicontrol(hFigure,'Position',[72 15 70 21],'String','Cancel','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@cancelCallback);
                uicontrol(hFigure,'Position',[164 15 70 21],'String','Ok','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@okCallback);
                uiwait(hFigure);
                if ~ishandle(hFigure), return;end
                if ~get(hFigure,'userData')
                    close(hFigure);
                    cobj = [];
                    return;
                end
                close(hFigure);
                drawnow
                val = g.GetPropertyValues();
                if length(val.startLatency) ~= length(val.endLatency), error('Latencies must have the same length.');end
                varargin{1} = [val.startLatency(:) val.endLatency(:)];
                varargin{2} = val.robustPcaFlag;
                varargin{3} = val.numberOfGaussianMixtures;
                dispCommand = true;
            end
            
            if narg < 1, varargin{1} = obj.timeStamp([1 end]);end
            if ~ismatrix(varargin{1}), error('First argument must be a matrix (n-chunks x 2) with pointers to clean chunks of data.');end
            segmentObj = basicSegment(varargin{1});
            
            if narg < 2, varargin{2} = false; end
            if ~isnumeric(varargin{2}), error('Second argument must be a either logical or numeric type.');end
            robustPcaFlag = varargin{2};
            
            if robustPcaFlag && narg < 3 error('Third argument is the number of gaussians.');end
            if robustPcaFlag && narg == 3 && ~isnumeric(varargin{3}), error('Third argument is the number of gaussians.');end
            if robustPcaFlag && narg == 3, numberOfGaussianMixtures = varargin{3};end
            
            commandHistory.commandName = 'projectDataPCA';
            commandHistory.uuid = obj.uuid;
            commandHistory.varargin = varargin;
            cobj = obj.copyobj(commandHistory);
            
            if dispCommand
                disp('Running:');
                disp(['  ' cobj.history]);
            end
            
            indices = false(1,size(obj,1));
            for it=1:length(segmentObj.startLatency), indices = indices | (obj.timeStamp >= segmentObj.startLatency(it) & obj.timeStamp < segmentObj.endLatency(it));end
            indices = find(indices(:));
            data = obj.dataInXYZ(indices,:,:);
                        
            cobj.projectionMatrix =  zeros(3,3,cobj.numberOfChannels/length(cobj.componetsToProject));
            projectionMatrix = cobj.projectionMatrix;
            I = repmat({1:length(indices)},obj.numberOfChannels/3,1);
            try
                obj.initStatusbar(1,obj.numberOfChannels/3,'Computing PCA rotation matrix...')
                for it=1:obj.numberOfChannels/3
                    
                    %-- getting rid of outliers
                    if robustPcaFlag, I{it} = rmOutlier(data(:,:,it),numberOfGaussianMixtures);end
                    
                    %-- estimating canonical axes in 3D
                    [R1,pc] = princomp(data(I{it},:,it));
                    %-- estimating rotation on z-axis
                    [~,loc] = max(corr(pc,data(I{it},:,it)));
                    z_axis = 3;find(loc==3);
                    p_axis = setdiff(loc(1:2),z_axis);
                    p_axis = p_axis(1);
                    target = data(I{it},[p_axis z_axis],it);
                    target = bsxfun(@minus,target,mean(target));
                    [~,~,T] = procrustes(pc(:,1:2),target);
                    R2 = T.T;
                    R2(end+1,end+1) = 0; %#ok
                    R2(end,end) = 1;
                    %--
                    if sign(det(R2))==1
                        projectionMatrix(:,:,it) = R1/R2; % clockwise rotation
                    else
                        projectionMatrix(:,:,it) = R1*R2; % counter clock
                    end
                    obj.statusbar(it);
                end
                obj.initStatusbar(1,obj.numberOfChannels/3,'Correcting orientation...')
                for it=2:obj.numberOfChannels/3
                    if sign(det(projectionMatrix(:,:,it)))==1
                        % projectionMatrix(:,:,it) = inv(projectionMatrix(:,:,it)); % clockwise rotation
                        projectionMatrix(:,:,it) = projectionMatrix(:,:,1); % clockwise rotation
                    end
                    obj.statusbar(it);
                end
                cobj.projectionMatrix = projectionMatrix;
                obj.initStatusbar(1,obj.numberOfChannels/3,'Projecting data...')
                for it=1:obj.numberOfChannels/3
                    pc = data(I{it},:,it)*cobj.projectionMatrix(:,:,it);
                    pc = bsxfun(@minus,pc,mean(pc));
                    cobj.dataInXY(indices(I{it}),:,it) = pc(:,1:2);
                    obj.statusbar(it);
                end
            catch ME
                obj.statusbar(obj.numberOfChannels/3);
                try obj.container.deleteItem(obj.container.findItem(cobj.uuid));end  %#ok
                ME.rethrow;
            end
        end
        %%
        function cobj = projectDataOntoCanonicalAxes(obj,plane)
            if nargin < 2, error('MoBILAB:noArguments','First argument must be a the axes: ''xy'', ''xz'', ''yz''.');end
            if nargin > 1 || isnumeric(plane)
                prompt = {'Enter the axes (''xy'',''xz'',''yz'')'};
                dlg_title = 'Project data onto canonical axes';
                num_lines = 1;
                def = {'xz'};
                plane = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(plane), return;end
                
            end
            if ~ischar(plane), error('MoBILAB:noArguments','First argument must be a the axes: ''xy'', ''xz'', ''yz''.');end
            
            switch lower(plane)
                case 'xy'
                    R = [1 0 0;0 1 0;0 0 0];
                    keepComponents = [1 2];
                case 'xz'
                    R = [1 0 0;0 0 0;0 0 1];
                    keepComponents = [1 3];
                case 'yz'
                    R = [0 0 0;0 1 0;0 0 1];
                    keepComponents = [2 3];
                otherwise
                    error('MoBILAB:noArguments','First argument must be a the axes: ''xy'', ''xz'', ''yz''.');
            end
            cobj = projectData(obj,R,keepComponents);
        end
        %%
        function cobj = projectData(obj,R,keepComponents)
            if nargin < 2, error('MoBILAB:noArguments','First argument must be a rotation matrix');end
            if ~ismatrix(R), error('MoBILAB:noArguments','First argument must be a rotation matrix');end
            R = reshape(R,[3 3 length(R)/9]);
            if nargin < 3, keepComponents = 1:2;end
            if keepComponents < 1 || keepComponents > 3 || length(keepComponents) ~=2, keepComponents = 1:2;end
            
            commandHistory.commandName = 'projectData';
            commandHistory.uuid = obj.uuid;
            commandHistory.varargin{1} = R;
            commandHistory.varargin{2} = keepComponents;
            cobj = copyobj(commandHistory);
            
            for it=1:obj.numberOfChannels/3
                pc = obj.dataInXYZ(:,:,it)*cobj.projectionMatrix(:,:,it);
                pc = bsxfun(@minus,pc,mean(pc));
                cobj.dataInXY(I{it},:,it) = pc(:,keepComponents);
            end
        end
        %%
        function epochObj = epoching(obj,eventLabelOrLatency, timeLimits, channels, condition,subjectID)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 5, condition = 'unknown';end
            if nargin < 6, subjectID = obj.uuid;end
           
            children = obj.children;
            if isempty(children)
               epochObj = epoching@dataStream(obj,eventLabelOrLatency, timeLimits, channels, condition);
               return;
            end
            [xy,time,eventInterval] = epoching@coreStreamObject(obj,eventLabelOrLatency, timeLimits, channels);
            [n,m,p] = size(xy);
            xy = squeeze(mean(xy,2));
            children = obj.children;
            nc = length(children);
            if ~isempty(children)
                derivativeLabel = cell(nc,1);
                data = zeros(n,p*nc,m);
                indices = reshape(1:p*nc,[p nc]);
                for it=1:nc
                    data(:,indices(:,it),:) = permute( epoching@coreStreamObject(children{it},eventLabelOrLatency, timeLimits, channels), [1 3 2] );
                    if ~isempty(strfind(children{it}.name,'vel')),       derivativeLabel{it} = 'Velocity';
                    elseif ~isempty(strfind(children{it}.name,'acc')),   derivativeLabel{it} = 'Acceleration';
                    elseif ~isempty(strfind(children{it}.name,'jerk')),  derivativeLabel{it} = 'Jerk';
                    else                                                 derivativeLabel{it} = ['Dt' num2str(it)];
                    end
                end
                pdata = zeros([n,nc,m]);
                for it=2:nc, pdata(:,it,:) = projectAB(data(:,indices(:,it),:),data(:,indices(:,1),:));end
                pdata(:,1,:) = sqrt(sum(data(:,indices(:,1),:).^2,2));
            else pdata = [];
            end
            epochObj = mocapEpoch(pdata,time,obj.label(channels),condition,eventInterval,subjectID,xy,derivativeLabel);
        end
        function epochObj = epochingTW(obj,latency, channels, condition, subjectID)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 4, condition = 'unknownCondition';end
            if nargin < 5, subjectID = obj.uuid;end
            if isa(obj,'mocap'), I = reshape(1:obj.numberOfChannels,[2 obj.numberOfChannels/2]);
            else I = reshape(1:obj.numberOfChannels,[3 obj.numberOfChannels/3]);
            end
            channelLabel = cell(length(channels),1);
            for it=1:length(channels), channelLabel{it} = num2str(channels(it));end
            channels = I(:,channels);
            channels = channels(:);
            [xy,~,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
            [n,m,p] = size(xy);
            xy = squeeze(mean(xy,2));
            children = obj.children;
            nc = length(children);
            if ~isempty(children)
                derivativeLabel = cell(nc,1);
                data = zeros(n,p*nc,m);
                indices = reshape(1:p*nc,[p nc]);
                for it=1:nc
                    data(:,indices(:,it),:) = permute( epochingTW@coreStreamObject(children{it},latency, channels), [1 3 2] );
                    if ~isempty(strfind(children{it}.name,'vel')),       derivativeLabel{it} = 'Velocity';
                    elseif ~isempty(strfind(children{it}.name,'acc')),   derivativeLabel{it} = 'Acceleration';
                    elseif ~isempty(strfind(children{it}.name,'jerk')),  derivativeLabel{it} = 'Jerk';
                    else                                                 derivativeLabel{it} = ['Dt' num2str(it)];
                    end
                end
                pdata = zeros([n,nc,m]);
                for it=2:nc, pdata(:,it,:) = projectAB(data(:,indices(:,it),:),data(:,indices(:,1),:));end
                pdata(:,1,:) = sqrt(sum(data(:,indices(:,1),:).^2,2));
            else pdata = [];
            end
            if size(latency,1) > 1, dl = round(mean(diff(latency,[],2))); else dl = diff(latency,[],2);end
            dlsum = cumsum(dl);
            if length(dl) > 3
                time = -fliplr(0:dlsum(2))/obj.samplingRate;
                time(end) = [];
                time = [time (0:dlsum(end) - dlsum(2)-1)/obj.samplingRate];
            else
                time = -fliplr(0:dlsum(1))/obj.samplingRate;
                time(end) = [];
                time = [time (0:dlsum(2) - dlsum(1)-1)/obj.samplingRate];
            end
            epochObj = mocapEpoch(pdata,time,channelLabel,condition,eventInterval,subjectID,xy,derivativeLabel);
        end
        %%
        function I = createEventsFromMagnitude(obj,varargin)
            % criteria: 'maxima', 'minima', 'zero crossing', '% maxima', '% minima', or 'all' (default: criteria = 'all')
            % channel: index of the channel where to search for the events
            
            dispCommand = false;
            if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
                prefObj = [...
                    PropertyGridField('channel',1,'DisplayName','Marker','Category','Main','Description','Mocap marker.')...
                    PropertyGridField('criteria','maxima','Type',PropertyType('char', 'row', {'maxima', 'minima','zero crossing'}),'DisplayName','Criteria','Category','Main','Description','Criterion for making the event, could be: maxima, minima, zero crossing.')...
                    PropertyGridField('eventType','max','DisplayName','Marker name','Category','Main','Description','Name of the new event marker.')...
                    ];

                hFigure = figure('MenuBar','none','Name','Create event marker','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
                    'Resize','off','userData',0);
                position = get(hFigure,'position');
                set(hFigure,'position',[position(1:2) 303 431]);
                hPanel = uipanel(hFigure,'Title','','BackgroundColor','white','Units','pixels','Position',[0 55 303 380],'BorderType','none');
                g = PropertyGrid(hPanel,'Properties', prefObj,'Position', [0 0 1 1]);%,'Description','Projects low-dimensional burst artifacts out of the data.');
                uicontrol(hFigure,'Position',[72 15 70 21],'String','Cancel','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@cancelCallback);
                uicontrol(hFigure,'Position',[164 15 70 21],'String','Ok','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@okCallback);
                uiwait(hFigure);
                if ~ishandle(hFigure), return;end
                if ~get(hFigure,'userData'), close(hFigure);return;end
                close(hFigure);
                drawnow
                val = g.GetPropertyValues();
                varargin{1} = val.channel;
                varargin{2} = val.criteria;
                varargin{3} = val.eventType;
                dispCommand = true;
            end
            
            Narg = length(varargin);
            if Narg < 1, channel   = 1;       else channel   = varargin{1}(1);end
            if Narg < 2, criteria  = 'maxima';else criteria  = varargin{2};   end
            if Narg < 3, eventType = [];else eventType = varargin{3};   end
            if Narg < 4
                inhibitedWindowLength = obj.samplingRate;
            else
                inhibitedWindowLength = ceil(obj.samplingRate*varargin{4});
            end
            if Narg < 5
                segmentObj = basicSegment([obj.timeStamp(1),obj.timeStamp(end)]);
            else
                segmentObj = varargin{5};
            end
            
            if dispCommand
                disp('Running:');
                disp('  latency = createEventsFromMagnitude(obj,mocap_marker, criteria, event_marker)');
            end
            
            numberOfSegments = length(segmentObj.startLatency);
            index = obj.getTimeIndex([segmentObj.startLatency segmentObj.endLatency]);
            index = reshape(index,numberOfSegments,2);
            
            if isempty(eventType)
                switch criteria
                    case 'maxima',        eventType = 'max';
                    case 'zero crossing', eventType = 'zc';
                    case 'minima',        eventType = 'min';
                    otherwise,            eventType = 'noname';
                end
            end
            signal = obj.magnitude(:,channel);
            
            for it=1:numberOfSegments
                I = searchInSegment(signal(index(it,1):index(it,2)),criteria,inhibitedWindowLength);
                time = obj.timeStamp(index(it,1):index(it,2));
                latency = obj.getTimeIndex(time(I));
                obj.event = obj.event.addEvent(latency,eventType);
            end
            if dispCommand, disp('Done.');end
        end
    end
    
    methods(Hidden=true)
        %
        function cobj = smoothDerivative(obj,varargin)
            warning('This method will be deprecated, instead use ''timeDerivative'' with the same input arguments.');
            cobj = timeDerivative(obj,varargin);
        end
        %%
        function cobj = bodyModelSetupFile(obj,filename)
            if nargin < 2
                [filename, pathname] = uigetfile2('*.xls','Select the file containing marker labels',...
                    obj.container.container.preferences.mocap.markerLabels);
                if isnumeric(filename) || isnumeric(pathname), return;end
                filename = fullfile(pathname,filename);
            end
            if ~exist(filename,'file'), error('MoBILAB:fileDoesNotExist','The file does not exist.');end
            [~,~,~,~,iG,segment] = readSegmentTable_xls(filename);
            obj.lsMarker2JointMapping = iG;
            indMarkes = segment(:,3) ~= 0;
            markers = unique(segment(indMarkes,3));
            markers = markers(1):markers(end);
            nmarkers = length(markers);
            count = 1;
            for it=1:(nmarkers*3):obj.numberOfChannels
                indices = it:it+(nmarkers*3)-1;
                cobj = divideStreamObject(obj,indices,obj.label(indices),[obj.name '_' num2str(count)]);
                cobj.bodyModelFile = filename;
                count = count+1;
            end
        end
        %%
        function cobj = inverseKinematics(obj)
            if ~exist(obj.bodyModelFile,'file'), error('You must have a body model file first, run bodyModelSetupFile(obj,xlsfile).');end
            
            commandHistory.commandName = 'inverseKinematics';
            commandHistory.uuid = obj.uuid;
            obj.container.container.lockGui('Adding an object to the tree. Please wait...');
            newHeader = createHeader(obj,commandHistory);
            obj.container.container.lockGui;
            if isempty(newHeader), error('Cannot make a copy of this object. Please provide a valid ''command history'' instruction.');end
            cobj = obj.container.addItem(newHeader);
            
            %-- Initialization with the least squares fit
            data   =  obj.mmfObj.Data.x;
            data_i = cobj.mmfObj.Data.x;
            I = data ~= 0;
            
            chz = find(100*sum(~I)/size(obj,1) > 50);
            
            hwait = waitbar(0);
            for it=1:size(obj,1)
                W = full(obj.lsMarker2JointMapping(I(it,:),:));
                W = bsxfun(@rdivide,W,sum(W));
                data_i(it,:) = data(it,I(it,:))*W;
                waitbar(it/size(obj,1),hwait);
            end
            close(hwait);
            cobj.mmfObj.Data.x = data_i;
            % cobj.mmfObj.Data.x = obj.mmfObj.Data.x*obj.lsMarker2JointMapping;
            %--
            
            [~,~,~,~,~,segment] = readSegmentTable_xls(obj.bodyModelFile);
            indices = segment(segment(:,3)~=0,3);
            Y = obj.dataInXYZ(:,:,indices);
            Y(Y==0) = NaN;
            Y = reshape(Y,[size(Y,1) 3*size(Y,3)]);
            X0 = initSS(obj,cobj,segment);
            
            dt = 1/cobj.samplingRate;
            datS = [3,20];                          % [w,p] position init. stdev
            datR = [0,5];                           % [w,p] position drift stdev
            datV = 1;                               % position sensor noise stdev
            ratio = 10;                             % position/angle stdev ratio

            % Prepare model for estimation
            [map, info, S, R, V] = prepare(segment, 1, dt,datS,datR,datV,ratio);
            
            
            xTrue = cobj.mmfObj.Data.x(1,:)';
            X = estimate(Y, X0, S, segment, map, info, R, V);
            cobj.mmfObj.Data.x = X;
        end
        %%    
        function cobj = kalmanFilter(obj,order,frameSize)
            if nargin < 2, order = 3;end
            if nargin < 3, frameSize = 128;end
            frameSize = 2*round(frameSize/2)+1;
            order = max(order);
            if ~exist(obj.bodyModelFile,'file'), error('You must have a body model file first, run bodyModelSetupFile(obj,xlsfile).');end
            commandHistory.commandName = 'kalmanFilter';
            commandHistory.uuid = obj.uuid;
            commandHistory.varargin = {order, frameSize};
            itemIndex = obj.container.findItem(obj.uuid);
            descendants = obj.container.getDescendants(itemIndex);
            cobj = cell(order+1,1); 
            
            if isempty(descendants)
                obj.initStatusbar(1,order+1,'Creating binary files...');
                newHeader = createHeader(obj,commandHistory);
                cobj{1} = bodyStream(newHeader);
                cobj{1}.container = obj.container;
                obj.container.item{end+1} = cobj{1};
                obj.statusbar(1);
                metadata = load(cobj{1}.header,'-mat');
                pathname = fileparts(cobj{1}.binFile);
                
                for it=2:order+1    
                    if ismac, [~,hash] = system('uuidgen'); else hash =  java.util.UUID.randomUUID;end
                    metadata.uuid = char(hash);
                    if it == 2, prename = 'vel_';
                    elseif it == 3, prename = 'acc_';
                    elseif it == 4, prename = 'jerk_';
                    else prename = ['Dt' num2str(it) '_'];
                    end
                    metadata.name = [prename cobj{1}.name];
                    metadata.binFile = fullfile(pathname,[metadata.name '_' metadata.uuid '.bin']);
                    copyfile(cobj{1}.binFile,metadata.binFile);
                    metadata.header = fullfile(pathname,[metadata.name '_' metadata.uuid '.hdr']);
                    newHeader = metadata2headerFile(metadata);
                    cobj{it} = bodyStream(newHeader);
                    cobj{it}.container = obj.container;
                    obj.container.item{end+1} = cobj{it};
                    obj.statusbar(it);
                end
            else for it=1:order+1, cobj{it} = obj.container.item{descendants(it)};end
            end
            for jt=1:order+1, cobj{jt}.mmfObj.Writable = true;end
            dt = 1/obj.samplingRate;
            load(obj.bodyModelFile);
            
            % Initialize state transition matrix
            F = triu(cobj{1}.connectivity) + eye(length(cobj{1}.nodeLabel));
            F = sparse(F);
            F = kron(F,eye(3));
            iSigmaF = F*F';
            % F = kron(eye((order+1)),F);       % A = kron(A,diag([1; dt*ones(order,1)]));
            
            F = eye(cobj{1}.numberOfChannels*(order+1)) + dt*diag(ones(1,cobj{1}.numberOfChannels*(order)),cobj{1}.numberOfChannels);
            
            Q = speye(size(F,1));            
            I = speye(obj.numberOfChannels);
            R = 1*I;
            Nst = (order+1)*cobj{1}.numberOfChannels;
            Nst_xyz = cobj{1}.numberOfChannels;
            N = size(obj,1);
            One = ones(frameSize,1);
            z = obj.mmfObj.Data.x;     % measurements
            
            for it=1:obj.numberOfChannels
                indZeros = find(z(:,it) == 0);
                if length(indZeros) < N
                    ind = setdiff(1:N,indZeros);
                    z(indZeros,it) = interp1(ind,z(ind,it),indZeros,'pchip');
                end
            end
            
            % Initial state conditions
            I = 1:frameSize;   % I = 10*obj.samplingRate:5:N-10*obj.samplingRate;        
            x_est = z(I,:)*iG;
            % F = (A*A')*x_est(2:end,:)'*x_est(1:end-1,:)*pinv(x_est(1:end-1,:)'*x_est(1:end-1,:));
            
            % Initialize measurement matrix  
            iSc = iG*iG';
            C = iSc*z(I,:)'*x_est*pinv((x_est'*x_est));
            offset = z(I,:)' - C*x_est';
            offset = median(offset,2);          
            G = kron([1 zeros(1,order)],C);
            G = [G offset];
            
            x_est = kron([1 zeros(1,order)],x_est)';
            P_est = eye(Nst);  
                 
            pcount = 1;
            t = frameSize+1;
            h = 1;
            obj.initStatusbar(frameSize-1,N-frameSize,'Filtering...')
            bObj = cobj{1}.plot;
            bObj2 = obj.plot;
            while t+frameSize <= N-frameSize
                obj.statusbar(t);    
                win = t-frameSize+1:t;
                
                % Predicted state and covariance
                x_prd = F * x_est;
                P_prd = F * P_est * F' + Q;
                
                % Estimated Kalman gain matrix
                B = G(:,1:end-1) * P_prd';
                S = B * G(:,1:end-1)' + R;
                K = (S \ B)';
                
                % Estimated state and covariance
                e = (z(win,:) - [x_prd' One]*G')';
                x_est = x_prd + K * e;
                P_est = P_prd - K/S*K';
                
                % Parameter estimation
                % if ~mod(pcount,100)
                %     F = iSigmaF*x_est(1:Nst_xyz,2:end)*x_est(1:Nst_xyz,1:end-1)'*pinv(x_est(1:Nst_xyz,1:end-1)*x_est(1:Nst_xyz,1:end-1)');
                %     F = kron(eye((order+1)),F);
                % end
                % C = iSc*z(win,:)'*x_est(1:Nst_xyz,:)'*pinv((x_est(1:Nst_xyz,:)*x_est(1:Nst_xyz,:)'));
                % offset = z(win,:)' - G(:,1:Nst_xyz)*x_est(1:Nst_xyz,:);
                % offset = mean(offset,2);
                % G(:,end) = offset;
                
                z_est = [mean(x_est,2)' 1]*G';
                z(t,:) = z_est;
                
                % streaming to the files
                tmp = reshape(mean(x_est,2),[cobj{1}.numberOfChannels,order+1]);
                for jt=1:order+1, cobj{jt}.mmfObj.Data.x(t,:) = tmp(:,jt);end
                
                if ~mod(pcount,100)
                    bObj.plotThisTimeStamp((t-1)/512);
                    bObj2.plotThisTimeStamp((t-1)/512);
                    drawnow;
                end
                
                t = t+h;
                pcount = pcount+1;
            end
            obj.statusbar(N);
            for jt=1:order+1, cobj{jt}.mmfObj.Writable = false;end
        end
        %%
        function properyArray = getPropertyGridField(obj)
            dim = size(obj.dataInXYZ);
            if length(dim) < 3, dim(3) = 1;end
            properyArray1 = getPropertyGridField@coreStreamObject(obj);
            properyArray2 = [...
                PropertyGridField('limits',obj.animationParameters.limits,'DisplayName','animationParameters.limits','ReadOnly',false,'Description','Room limits.')...
                PropertyGridField('conn',obj.animationParameters.conn,'DisplayName','animationParameters.conn','ReadOnly',false,'Description','Connection matrix.')...
                PropertyGridField('dataInXYZ',['<' num2str(dim(1)) 'x' num2str(dim(2)) 'x' num2str(dim(3)) ' ' obj.precision '>'],'DisplayName','dataInXYZ','ReadOnly',false,'Description','DataInXYZ is a dependent property, for intensive I/O operations is more efficient accessing directly the memory mapped file object, obj.mmfObj.Data.x.')...
                PropertyGridField('magnitude',['<' num2str(dim(1)) 'x' num2str(dim(3)) ' ' obj.precision '>'],'DisplayName','magnitude','ReadOnly',false,'Description','Magnitud of the vectors in dataInXYZ.')...
                ];
            properyArray = [properyArray1 properyArray2];
        end
        %%
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid        = obj.uuid;
            end
            newHeader = createHeader@dataStream(obj,commandHistory);
            if ~isempty(newHeader), return;end
            
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            uuid = generateUUID;
            metadata.uuid = uuid;
            path = fileparts(obj.binFile);
            
            switch commandHistory.commandName
                case 'removeOcclusionArtifact'
                    prename = 'remOcc_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    channels = commandHistory.varargin{2};
                    metadata.numberOfChannels = length(channels);
                    metadata.label = obj.label(channels);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'timeDerivative'
                    order = commandHistory.varargin{1};
                    if ~isnumeric(order), error('Input parameter ''order'' must be a number indicating the order of the time derivative to calculate.');end
                    switch order
                        case 1, prename = 'vel_';
                        case 2, prename = 'acc_';
                        case 3, prename = 'jerk_';
                        otherwise
                            prename = ['Dt' num2str(order) '_'];
                    end
                    index = obj.container.findItem(commandHistory.uuid);
                    parentName = obj.container.item{ index }.name;
                    metadata.name = [prename parentName];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    if order == 1
                        channels = commandHistory.varargin{3};
                        fid = fopen(metadata.binFile,'w');
                        fwrite(fid,obj.mmfObj.Data.x(:,channels),obj.precision);
                        fclose(fid);
                        metadata.numberOfChannels = length(channels);
                        metadata.label = obj.label(channels);
                    else
                        copyfile(obj.binFile,metadata.binFile,'f');
                    end
                    
                case 'kalmanFilter'
                    prename = 'body_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    load(obj.bodyModelFile);
                    metadata.numberOfChannels = length(nodeLabel)*3;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    metadata.connectivity = connectivity; %#ok
                    metadata.nodeLabel = nodeLabel;
                    newLabel = repmat(nodeLabel,1,3);
                    for it=1:size(newLabel,1), newLabel{it,1} = [newLabel{it,1} '_x'];end
                    for it=1:size(newLabel,1), newLabel{it,2} = [newLabel{it,2} '_y'];end
                    for it=1:size(newLabel,1), newLabel{it,3} = [newLabel{it,3} '_z'];end
                    newLabel = newLabel';
                    metadata.label = newLabel(:);
                    
                case 'inverseKinematics'
                    prename = 'ik_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    [treeNodes,kinematicTree,nodes,connectivity,ls_marker2nodesMapping] = readSegmentTable_xls(obj.bodyModelFile);
                    metadata.connectivity = connectivity;
                    metadata.nodes = nodes;
                    metadata.kinematicTree = kinematicTree;
                    metadata.treeNodes = treeNodes;
                    metadata.lsMarker2JointMapping = ls_marker2nodesMapping;
                    metadata.numberOfChannels = length(nodes)*3;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                    newLabel = repmat(nodes,1,3);
                    for it=1:size(newLabel,1), newLabel{it,1} = [newLabel{it,1} '_x'];end
                    for it=1:size(newLabel,1), newLabel{it,2} = [newLabel{it,2} '_y'];end
                    for it=1:size(newLabel,1), newLabel{it,3} = [newLabel{it,3} '_z'];end
                    newLabel = newLabel';
                    metadata.label = newLabel(:);
                    
                case 'save_as'
                    prename = '';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    uind = unique(obj.animationParameters.conn(:));
                    [~,loc_i] = ismember(obj.animationParameters.conn(:,1),uind) ;
                    [~,loc_j] = ismember(obj.animationParameters.conn(:,2),uind) ;
                    obj.reshape([length(obj.timeStamp) 3 obj.numberOfChannels/3]);
                    data = obj.mmfObj.Data.x(:,:,uind);
                    obj.reshape([length(obj.timeStamp) obj.numberOfChannels]);
                    metadata.animationParameters = obj.animationParameters;
                    metadata.animationParameters.conn = [loc_i loc_j];
                    tmp = uind*3;
                    channels = [];
                    for it=1:length(tmp)
                        channels(end+1) = tmp(it)-2;%#ok
                        channels(end+1) = tmp(it)-1;%#ok
                        channels(end+1) = tmp(it);%#ok
                    end
                    
                    fid = fopen(metadata.binFile,'w');
                    if fid<=0, error('Invalid file identifier. Cannot create the copy object.');end;
                    fwrite(data(:),obj.precision);
                    fclose(fid);
                    N = length(channels);
                    metadata.label = cell(N,1);
                    for it=1:N, metadata.label{it} = obj.label{channels(it)};end
                    
                case {'projectData' 'projectDataPCA'}
                    prename = 'pca_';
                    metadata.name = [prename obj.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = obj.numberOfChannels*2/3;
                    metadata.class = 'pcaMocap';
                    I = [1:3:obj.numberOfChannels;2:3:obj.numberOfChannels];
                    metadata.label = obj.label(I(:));
                    Zeros = zeros(length(metadata.timeStamp),1);
                    fid = fopen(metadata.binFile,'w');
                    if fid<=0, error('Invalid file identifier. Cannot create the copy object.');end;
                    for it=1:metadata.numberOfChannels, fwrite(Zeros,obj.precision);end
                    fclose(fid);
                    
                otherwise
                    error('Cannot make a secure copy of this object. Please provide a valid ''commandHistory'' instruction.');
            end
            newHeader = metadata2headerFile(metadata);
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Add stick figure');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'loadConnectedBody',-1});
            jmenu.add(menuItem);         
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Filling-in occluded time points');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'removeOcclusionArtifact',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Lowpass filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'lowpass',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Compute time derivatives');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'timeDerivative',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot stick figure');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'mocapBrowser',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Time frequency analysis (CWT)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'continuousWaveletTransform',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Create event marker');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'createEventsFromMagnitude',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Inspect');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'inspect',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Annotation');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('Generate batch script');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@generateBatch_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="maroon">Delete object</HTML>');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});
        end
    end
end

%%
function okCallback(hObject,~,~)
set(get(hObject,'parent'),'userData',1);
uiresume;
end
%%
function cancelCallback(hObject,~,~)
set(get(hObject,'parent'),'userData',0);
uiresume;
end