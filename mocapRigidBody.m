%% mocapRigidBody class
% Creates and uses a mocap object using predefined rigidbodies in the mocap data.
%
% Author: Marius Klug, TU Berlin, 13-Sep-2016 adapted from Alejandro Ojeda

%%
classdef mocapRigidBody < dataStream
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
        function obj = mocapRigidBody(header)
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
            metadata.artifactMask = sum(metadata.artifactMask(:) ~= 0);
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
        function cobj = throwOutChannels(obj,varargin)
            % Throws out selected channels.
            %
            % Input arguments:
            %       channels
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.throwOutChannels(channelsToThrow);
            %
            
            dispCommand = false;
            
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1}(1) == -1
                prompt = {'Enter channel numbers to throw out.'};
                dlg_title = 'Input parameters';
                num_lines = 1;
                channelsToThrow = inputdlg2(prompt,dlg_title,num_lines,{''});
                varargin{1} = str2num(channelsToThrow{1});
                if isempty(varargin), return;end
                dispCommand = true;
            end
            
            commandHistory.commandName = 'throwOutChannels';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = varargin{1};
            cobj = obj.copyobj(commandHistory);
            
            if dispCommand
                disp('Running:');
                disp(['  ' cobj.history]);
            end
            
            channelsToThrow = commandHistory.varargin{1};
            channelsToKeep = 1:obj.numberOfChannels;
            channelsToKeep(channelsToThrow) = [];
            
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x(:,channelsToKeep);
            
        end
        
        %%
        function cobj = addChannels(obj,varargin)
            % Adds channels to the data stream. Channels will be appended automatically at the end.
            %
            % Input arguments:
            %       channelLabels: A cell containing strings for labels of the additional channels
            %       data: a matrix for the new channels.
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.addChannels(channelLabels, dataMatrix);
            
            dispCommand = false;
            
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                error('Dont use the GUI for this, its not implemented.');
            else
                if length(varargin) ~= 2
                    error('Enter two arguments: The number of channels to add and the data matrix of them!')
                end
                
            end
            
            allData = obj.mmfObj.Data.x;
            allData(:,end+1:end+size(varargin{2},2)) = varargin{2};
            
            allLabels = obj.label;
            allLabels(end+1:end+length(varargin{1})) = varargin{1};
            
            commandHistory.commandName = 'addChannels';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = varargin{1};
            commandHistory.varargin{2} = 'This would be the data matrix of the added channels.';
            
            
            cobj = obj.copyobj(commandHistory);
            
            cobj.mmfObj.Data.x = allData;
            
        end
        
        %%
        function cobj = degToRad(obj,varargin)
            % Converts data channels from degree to radian (rad = deg / 180 * pi), mostly for better visualization in the same plot since the
            % scale is similar to the position then.
            %
            % Input arguments:
            %       channels
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.degToRad(channelsToConvert);
            
            
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            dispCommand = false;
            
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                
                prompt = {'Enter channels to convert!'};
                dlg_title = 'Input parameters';
                num_lines = 1;
                def = {''};
                channelsToConvert = inputdlg2(prompt,dlg_title,num_lines,def);
                varargin{1} = str2num(channelsToConvert{1});
                if isempty(varargin), return;end
                dispCommand = true;
                
            end
            
            
            
            commandHistory.commandName = 'degToRad';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = varargin{1};
            cobj = obj.copyobj(commandHistory);
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x;
            
            if dispCommand
                disp('Running:');
                disp(['  ' cobj.history]);
            end
            
            cobj.mmfObj.Data.x(:,varargin{1}) = cobj.mmfObj.Data.x(:,varargin{1}) / 180 * pi;
            
        end
        
        %%
        function cobj = unflipSigns(obj,varargin)
            % Heuristic for unflipping the sign of quaternion values to anable filtering. Quaternions can represent the
            % same value two ways, whereas only the sign changes. Sometimes the representation flips in the time series.
            % If this gets filtered, it creates an artifact, so this is why we want to unflip it first. It won't make a
            % difference later when transforming the values to Euler angles.
            % The principle idea is to check if the difference between consecutive values becomes smaller if we flip them.
            %
            % Input arguments:
            %       none.
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.unflipSigns();
            
            for channel = 1:obj.numberOfChannels
                
                % checking for already present eulers
                if ~isempty(strfind(obj.label{channel},'Euler'))
                    error('You can only unflip Quaternions, try it with the original data set.')
                end
                
            end
            
            commandHistory.commandName = 'unflipSigns';
            commandHistory.uuid        = obj.uuid;
            cobj = obj.copyobj(commandHistory);
            data = obj.mmfObj.Data.x;
            
            
            disp('Running:');
            disp(['  ' cobj.history]);
            
            
            % find correct channelnumber for the quaternion values of
            % this RB
            quaternionX = ~cellfun(@isempty,strfind(lower(obj.label),'quat_x'));
            quaternionY = ~cellfun(@isempty,strfind(lower(obj.label),'quat_y'));
            quaternionZ = ~cellfun(@isempty,strfind(lower(obj.label),'quat_z'));
            quaternionW = ~cellfun(@isempty,strfind(lower(obj.label),'quat_w'));
            
            % take the values
            X = data(:,quaternionX);
            Y = data(:,quaternionY);
            Z = data(:,quaternionZ);
            W = data(:,quaternionW);
            
            
            for dataPoint = 5:size(data,1)
                
                epsilon = 0.5;
                
                Adiff = abs(X(dataPoint-1) - X(dataPoint));
                Adiff2 = abs(X(dataPoint-2) - X(dataPoint-1));
                Adiff3 = abs(X(dataPoint-3) - X(dataPoint-2));
                Adiff4 = abs(X(dataPoint-4) - X(dataPoint-3));
                AsumOfPreviousDiffs = Adiff2 + Adiff3 + Adiff4;
                
                AflippedDataPoint = -X(dataPoint);
                
                AdiffFlipped = abs(X(dataPoint-1) - AflippedDataPoint);
                
                AconditionMet = AdiffFlipped<Adiff;% & Adiff > AsumOfPreviousDiffs;
                AbigJump = AdiffFlipped> 0.5;
                
                Bdiff = abs(Y(dataPoint-1) - Y(dataPoint));
                Bdiff2 = abs(Y(dataPoint-2) - Y(dataPoint-1));
                Bdiff3 = abs(Y(dataPoint-3) - Y(dataPoint-2));
                Bdiff4 = abs(Y(dataPoint-4) - Y(dataPoint-3));
                BsumOfPreviousDiffs = Bdiff2 + Bdiff3 + Bdiff4;
                
                BflippedDataPoint = -Y(dataPoint);
                
                BdiffFlipped = abs(Y(dataPoint-1) - BflippedDataPoint);
                
                BconditionMet = BdiffFlipped<Bdiff;% & Bdiff > BsumOfPreviousDiffs;
                BbigJump = BdiffFlipped> epsilon;
                
                Cdiff = abs(Z(dataPoint-1) - Z(dataPoint));
                Cdiff2 = abs(Z(dataPoint-2) - Z(dataPoint-1));
                Cdiff3 = abs(Z(dataPoint-3) - Z(dataPoint-2));
                Cdiff4 = abs(Z(dataPoint-4) - Z(dataPoint-3));
                CsumOfPreviousDiffs = Cdiff2 + Cdiff3 + Cdiff4;
                
                CflippedDataPoint = -Z(dataPoint);
                
                CdiffFlipped = abs(Z(dataPoint-1) - CflippedDataPoint);
                
                CconditionMet = CdiffFlipped<Cdiff;% & Cdiff > CsumOfPreviousDiffs;
                CbigJump = CdiffFlipped> epsilon;
                
                Ddiff = abs(W(dataPoint-1) - W(dataPoint));
                Ddiff2 = abs(W(dataPoint-2) - W(dataPoint-1));
                Ddiff3 = abs(W(dataPoint-3) - W(dataPoint-2));
                Ddiff4 = abs(W(dataPoint-4) - W(dataPoint-3));
                DsumOfPreviousDiffs = Ddiff2 + Ddiff3 + Ddiff4;
                
                DflippedDataPoint = -W(dataPoint);
                
                DdiffFlipped = abs(W(dataPoint-1) - DflippedDataPoint);
                
                DconditionMet = DdiffFlipped<Ddiff;% & Ddiff > DsumOfPreviousDiffs;
                DbigJump = DdiffFlipped> epsilon;
                
                
                
                if (AconditionMet+BconditionMet+CconditionMet+DconditionMet >= 2 && AbigJump+BbigJump+CbigJump+DbigJump == 0)
                    
                    X(dataPoint) = -X(dataPoint);
                    Y(dataPoint) = -Y(dataPoint);
                    Z(dataPoint) = -Z(dataPoint);
                    W(dataPoint) = -W(dataPoint);
                    
                end
                
            end
            
            data(:,quaternionX) = X;
            data(:,quaternionY) = Y;
            data(:,quaternionZ) = Z;
            data(:,quaternionW) = W;
            
            %             end
            cobj.mmfObj.Data.x = data;
            
        end
        
        
        
        
        %%
        function cobj = switchCoordinateSystem(obj,varargin)
            % switches between a left- and right hand sided coordinate system. Necessary if your VR system or one of the
            % mocap systems has a flipped axis. Basically just flips the sign of left/right. Skip this if you don't have
            % any issues!
            %
            % Input arguments:
            %       none.
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.switchCoordinateSystem();
            
            
            % make new object
            commandHistory.commandName = 'switchCoordinateSystem';
            commandHistory.uuid        = obj.uuid;
            cobj = obj.copyobj(commandHistory);
            
            disp('Running:');
            disp(['  ' cobj.history]);
            
            
            data = obj.mmfObj.Data.x;
            
            % now fill with data
            for channel = 1:obj.numberOfChannels
                
                if ~isempty(strfind(lower(obj.label{channel}),'rigid_z')) ||...
                        ~isempty(strfind(lower(obj.label{channel}),'quat_x')) ||...
                        ~isempty(strfind(lower(obj.label{channel}),'quat_y')) ||...
                        ~isempty(strfind(lower(obj.label{channel}),'euler_yaw')) ||...
                        ~isempty(strfind(lower(obj.label{channel}),'euler_pitch'))
                    
                    data(:,channel) = data(:,channel) .* -1;
                    
                end
                
            end
            
            cobj.mmfObj.Data.x = data;
            
        end
        
        %%
        function cobj = quaternionsToEuler(obj,varargin)
            % Transforms Quaternion angle values (4 dimensions) into Euler angles (3 dimensions yaw, pitch, roll) to
            % make them human-interpretable
            %
            % Input arguments:
            %       none.
            %
            % Output argument:
            %       cobj: handle to the new object
            %
            % Usage:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       newMocapObj = mocapObj.quaternionsToEuler();
            
            for channel = 1:obj.numberOfChannels
                
                % checking for already present eulers
                if ~isempty(strfind(obj.label{channel},'Euler'))
                    error('Don''t try to convert to Euler twice... that''s kinda obvious, isn''t it?')
                end
                
            end
            
            
            % no eulers are present, therefore each RB has 7 channels:
            % XYZABCD, from which ABCD are the quaternion values
 
            data = obj.mmfObj.Data.x;
            newData = zeros(size(obj.mmfObj.Data.x,1),size(obj.mmfObj.Data.x,2) - 1);
            newLabel = cell(obj.numberOfChannels-1,1);
            % the new Euler data has 1 channel less than the quaternions
            
            % fill the new data set and its label with all initial position data
            newLabel(1:3) = obj.label(1:3);
            newData(:,1:3) = obj.mmfObj.Data.x(:,1:3);
            
            % fill also with additional data if there are more channels
            % than just 7
            
            numberOfExcessChannels = obj.numberOfChannels - 7;
            
            if numberOfExcessChannels > 0
                
                newLabel(7:(7+numberOfExcessChannels-1)) = obj.label(8:(7+numberOfExcessChannels));
                newData(:,7:(7+numberOfExcessChannels-1)) = obj.mmfObj.Data.x(:,8:(7+numberOfExcessChannels));
                
            end
            
            
            % now fill with euler data
            
            % find correct channelnumber for the quaternion values of
            % this RB
            quaternionX = ~cellfun(@isempty,strfind(lower(obj.label),'quat_x'));
            quaternionY = ~cellfun(@isempty,strfind(lower(obj.label),'quat_y'));
            quaternionZ = ~cellfun(@isempty,strfind(lower(obj.label),'quat_z'));
            quaternionW = ~cellfun(@isempty,strfind(lower(obj.label),'quat_w'));
            
            % take the values
            % The orientation axes' labels in PhaseSpace are different from the
            % ones in Wikipedia: Forward is z (Wikipedia x), sideways is x (Wikipedia y), upways is
            % y (Wikipedia z)
            z = data(:,quaternionX);
            x = data(:,quaternionY);
            y = data(:,quaternionZ);
            w = data(:,quaternionW);
            
            
            % check if values are [-1 1] - could have been messed up by
            % interpolating
            
            w(w>=1) = 0.99999;
            w(w<=-1) = -0.99999;
            x(x>=1) = 0.99999;
            x(x<=-1) = -0.99999;
            y(y>=1) = 0.99999;
            y(y<=-1) = -0.99999;
            z(z>=1) = 0.99999;
            z(z<=-1) = -0.99999;
            
            
            % transform those values to euler angles
            % see https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            % the rotation occurs in the order yaw, pitch, roll (about body-fixed axes).
            
            % now since we took the different axes for the quaternions
            % before, this has to be taken backwards to obtain the
            % correct values for yaw, pitch and roll
            % -> this means that roll (rotation about x) becomes yaw, pitch (rotation about y) becomes
            % roll and yaw (rotation about z) becomes pitch.
            
            % for some reason after filtering the quaternion values, it
            % can happen that the transformation below results in
            % complex numbers, whereas the real part is exactly pi (or
            % -pi) and the complex number has some kind of "excess"
            % value. I don't know why this happens, but I just take the
            % real part, since it's the most reasonable thing to do, I
            % guess...
            
            channelEulerYaw = real(atan2(2.*(w.*x + y.*z),1 - 2.*(x.^2 + y.^2)));     % wikipedia roll (rotation about x)
            channelEulerRoll = real(asin(2.*(w.*y - z.*x)));                             % wikipedia pitch (rotation about y)
            channelEulerPitch = real(atan2(2.*(w.*z + x.*y),1-2.*(y.^2+z.^2)));          % wikipedia yaw (rotation about z)
            
            % convert from radian to degree
            
            factor = 180/pi;
            
            channelEulerYaw = channelEulerYaw*factor;
            channelEulerRoll = channelEulerRoll*factor;
            channelEulerPitch = channelEulerPitch*factor;
            
            
            % actually fill new data set and labels
            
            newData(:,4) = channelEulerYaw;
            newData(:,5) = channelEulerPitch;
            newData(:,6) = channelEulerRoll;
            
            newLabel{4} = strcat(obj.label{4}(1:strfind(lower(obj.label{4}),'quat_x')-1),'Euler_Yaw');
            newLabel{5} = strcat(obj.label{4}(1:strfind(lower(obj.label{4}),'quat_x')-1),'Euler_Pitch');
            newLabel{6} = strcat(obj.label{4}(1:strfind(lower(obj.label{4}),'quat_x')-1),'Euler_Roll');
            
            % make new object
            commandHistory.commandName = 'quaternionsToEuler';
            commandHistory.uuid        = obj.uuid;
            commandHistory.newLabels = newLabel;
            cobj = obj.copyobj(commandHistory);
            
            disp('Running:');
            disp(['  ' cobj.history]);
            
            cobj.mmfObj.Data.x = newData;
            cobj.label = newLabel;
            
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
                    indi = data(:,channels(it))==0 | data(:,channels(it))==1; % one of the quaternion channels is 1 if occluded
                    
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
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(3)};
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
        function cobj = oneEuroFilter(obj, varargin)
            % Filters the motion capture data with a one euro filter.
            % See G?ry Casiez, Nicolas Roussel, Daniel Vogel.  1? Filter:  A Simple Speed-based Low-pass Filter for
            % Noisy Input in Interactive Systems.  CHI?12, the 30th Conference on Human Factors in Computing
            % Systems, May 2012, Austin, United States. ACM, pp.2527-2530, 2012, <10.1145/2207676.2208639>.
            % <hal-00670496>
            %
            % Input arguments:
            %       mincutoff:   minimum lowpass cutoff frequency (in Hz)
            %       beta: beta value for adaptation of the filter -> higher value lead to less lag and more jitter for
            %       high frequency data
            %
            % Output argument:
            %       cobj:      handle to the new object
            %
            % Usage:
            %       mincutoff    = 1;
            %       beta         = 2;
            %       mocapObjFilt = mobilab.allStreams.item{7}.oneEuroFilter( mincutoff , beta );
            
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter the minimal cutoff frequency:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(1.0)};
                inputFromDialog = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(inputFromDialog), return;end
                varargin{1} = str2double(inputFromDialog{1});
                if isnan(varargin{1}), return;end
                
                prompt = {'Enter the beta value:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                %def = {num2str(obj.container.container.preferences.mocap.lowpassCutoff)};
                def = {num2str(5.0)};
                inputFromDialog = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(inputFromDialog), return;end
                varargin{2} = str2double(inputFromDialog{1});
                if isnan(varargin{2}), return;end
                
                dispCommand = true;
            end
            
            % Cutoff Frequency and beta values
            if nargin < 1, mincutoff = 1.0; else mincutoff = varargin{1};end
            if nargin < 2, beta = 5.0; else beta = varargin{2};end
            
            try
                
                
                noisySignal = obj.mmfObj.Data.x;
                commandHistory.commandName = 'oneEuroFilter';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = mincutoff;
                commandHistory.varargin{2} = beta;
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                %Declare oneEuro object
                theOneEuroFilter = oneEuro;
                %Alter filter parameters to tune
                theOneEuroFilter.mincutoff = mincutoff;
                theOneEuroFilter.beta = beta;
                
                filteredSignal = zeros(size(noisySignal));
                
                obj.initStatusbar(1,size(noisySignal,2),'Filtering...');
                
                % filter all channels
                for channelToFilter = 1:size(noisySignal,2)
                    
                    % the filter goes through all data points
                    for dataPoint = 1:size(noisySignal,1)
                        filteredSignal(dataPoint,channelToFilter) = theOneEuroFilter.filter(noisySignal(dataPoint,channelToFilter),dataPoint);
                    end
                    
                    obj.statusbar(channelToFilter);
                    
                end
                
                % add the filtered data
                cobj.mmfObj.Data.x = filteredSignal;
                
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
            if nargin < 3, fc = 6;  else fc = varargin{2};end
            if nargin < 4, channels = 1:obj.numberOfChannels;else channels = varargin{3};end
            if nargin < 5, filterOrder = 128;else filterOrder = varargin{4};end
            if ~isnumeric(order), error('prog:input','First argument must be the order of the derivative (1=veloc, 2=acc, 3=jerk).');end
            if ~isnumeric(fc),    error('prog:input','Second argument must be the cut off frequency.');end
            if ~isnumeric(channels), error('Invalid channel.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid channel.');end
            
            % checking for Quaternionvalues
            if sum(~cellfun('isempty',strfind(obj.label,'_A')))>0, error('Please transform quaternion values to euler angles before calculating time derivatives!');end
            
            Nch = length(channels);
            dt = 1/obj.samplingRate;
            %             dt = 1e3*dt; % from seconds to mili seconds
            order = unique(1:max(order));
            N = max(order);
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
                tmpData = obj.mmfObj.Data.x;
                for derivative=1:N
                    for channel=1:size(tmpData,2)
                        
                        % deriving
                        tmpData(1:end-1,channel) = diff(tmpData(:,channel),1)/dt;
                        tmpData(end,channel) = tmpData(end-1,channel);
                        
                        % check if channel is Euler angles and if so,
                        % correct for turns over pi or -pi respectively
                        if strfind(obj.label{channel},'Euler')
                            
                            %                             cobj.mmfObj.Data.x(cobj.mmfObj.Data.x > 2*pi, jt) = cobj.mmfObj.Data.x(cobj.mmfObj.Data.x > 2*pi, jt) - 2*pi;
                            %                             cobj.mmfObj.Data.x(cobj.mmfObj.Data.x < -2*pi, jt) = cobj.mmfObj.Data.x(cobj.mmfObj.Data.x < -2*pi, jt) + 2*pi;
                            
                            dataChannel = tmpData(:,channel);
                            
                            % turning rates of more than half a circle per frame are not possible
                            dataChannel(dataChannel > 180/dt) = dataChannel(dataChannel > 180/dt) - 2*180/dt;
                            dataChannel(dataChannel < -180/dt) = dataChannel(dataChannel < -180/dt) + 2*180/dt;
                            
                            tmpData(:,channel) = dataChannel;
                            
                        end
                        
                        % smoothing
                        %                       %  cobj.mmfObj.Data.x(:,jt) = filtfilt_fast(b,a,cobj.mmfObj.Data.x(:,jt));
                        obj.statusbar((Nch*(derivative-1)+channel));
                    end
                    
                    % creating the new object and filling the derived data
                    
                    commandHistory.varargin{1} = derivative;
                    cobj = tmpObj.copyobj(commandHistory);
                    
                    cobj.mmfObj.Data.x = tmpData;
                    
                    
                    % soft masking
                    
                    if derivative==1, artifactIndices = cobj.artifactMask(:) ~= 0;end
                    if any(artifactIndices), cobj.mmfObj.Data.x(artifactIndices) = cobj.mmfObj.Data.x(artifactIndices).*(1-cobj.artifactMask(artifactIndices));end
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
        function addEventsFromDerivatives(obj,varargin)
            % Adds all events from time derivative streams (vel, acc, jerk, DtX) of this stream into this stream. Mostly
            % to be used for visually inspecting movement on and offset event markers
            %
            % Input arguments:
            %       none
            %
            % Output argument:
            %       none. The events will be saved in the present data set.
            %
            % Uses:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       mocapObj.addEventsFromDerivatives();

            
            disp('Running:');
            disp('obj.addEventsFromDerivatives()');
            
            for children = 1:size(obj.children,2)
                
                if ~isempty(strfind(obj.children{children}.name,'vel')) ||...
                        ~isempty(strfind(obj.children{children}.name,'acc')) ||...
                        ~isempty(strfind(obj.children{children}.name,'jerk')) ||...
                        ~isempty(strfind(obj.children{children}.name,'Dt'))
                    
                    derivativeEvent = obj.children{children}.event;
                    obj.event = obj.event.addEvent(derivativeEvent.latencyInFrame,derivativeEvent.label);
                    
                end
            end
            
        end
        
        %%
%         function movementOnsetsAndEnds(obj,varargin)
%             
%             dispCommand = false;
%             if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
%                 
%                 dispCommand = true;
%             end
%             
%             if dispCommand
%                 disp('Running:');
%                 disp('movementOnsetsAndEnds');
%             end
%             
%             events = obj.event;
%             maxPosition = ~cellfun(@isempty,strfind(events.label,'max'));
%             minPosition = ~cellfun(@isempty,strfind(events.label,'min'));
%             zeroCrossPosition = ~cellfun(@isempty,strfind(events.label,'zeroCross'));
%             
%             maxEvents=events;
%             minEvents=events;
%             
%             maxEvents.label(~(maxPosition+zeroCrossPosition))=[];
%             maxEvents.latencyInFrame(~(maxPosition+zeroCrossPosition))=[];
%             maxEvents.hedTag(~(maxPosition+zeroCrossPosition))=[];
%             
%             latenciesIndex = 1;
%             movementOnsetFound = 0;
%             for i = 1:size(maxEvents.label,1)-2
%                 if ~isempty(strfind(maxEvents.label{i},'zeroCrossVel')) && ~isempty(strfind(maxEvents.label{i+1},'Jerk')) && ~isempty(strfind(maxEvents.label{i+2},'Acc')) %&& ~isempty(strfind(maxEvents.label{i+2},'Vel'))
%                     movementOnsetFound = 1;
%                     latencies(latenciesIndex) = maxEvents.latencyInFrame(i+1);
%                     latenciesIndex = latenciesIndex + 1;
%                 end
%             end
%             
%             if movementOnsetFound
%                 events = events.addEvent(latencies,'movementOnset');
%             end
%             
%             minEvents.label(~(minPosition+zeroCrossPosition))=[];
%             minEvents.latencyInFrame(~(minPosition+zeroCrossPosition))=[];
%             minEvents.hedTag(~(minPosition+zeroCrossPosition))=[];
%             
%             latenciesIndex = 1;
%             for i = 1:size(minEvents.label,1)-2
%                 if ~isempty(strfind(minEvents.label{i},'zeroCrossVel')) && ~isempty(strfind(minEvents.label{i+1},'Jerk')) && ~isempty(strfind(minEvents.label{i+2},'Acc'))
%                     movementOnsetFound = 1;
%                     latencies(latenciesIndex) = minEvents.latencyInFrame(i+1);
%                     latenciesIndex = latenciesIndex + 1;
%                 end
%             end
%             
%             if movementOnsetFound
%                 events = events.addEvent(latencies,'movementOnset');
%             end
%             
%             obj.event = events;
%             
%             if dispCommand, disp('Done.');end
%         end
        
        %%
        function deleteMarkers(obj,varargin)
            % Removes event markers in the stream. Optionally keeps specified markers.
            %
            % Input arguments:
            %       markersToKeep:    (OPTIONAL) Cell array of markers that should be kept in the data stream. 
            %
            % Output argument:
            %       none. The events will be saved in the present data set.
            %
            % Uses:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       mocapObj.deleteMarkers();
            %       mocapObj.deleteMarkers(markersToKeep);
            
            dispCommand = false;
            
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Which markers to keep?'};
                dlg_title = 'Input parameters';
                num_lines = 1;
                def = {''};
                markersToKeep = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                markersToKeep = strsplit(markersToKeep{1}, ' ');
                dispCommand = true;
            else
                if ~isempty(varargin)
                    markersToKeep = varargin{1};
                else
                    markersToKeep = {''};
                end
            end
            
            command = 'Delete Markers, keep only:';
            for marker = 1:numel(markersToKeep)
                
                command = strcat(command, {' '}, markersToKeep(marker));
                
            end
            
            if dispCommand
                disp('Running:');
                disp(command);
            end
            
            % determine for each unique marker if it is of the markers to keep or not
            if ~isempty(obj.event.uniqueLabel)
                for uniqueLabelInd = 1:size(obj.event.uniqueLabel,1)
                    for markerInd = 1:numel(markersToKeep)

                        comparison(uniqueLabelInd,markerInd) = strcmp(obj.event.uniqueLabel{uniqueLabelInd},markersToKeep{markerInd});

                    end
                end
                comparison = sum(comparison,2);

                tempEvents = obj.event;
                % delete markers
                for uniqueLabelInd = 1:size(obj.event.uniqueLabel,1)

                    if ~comparison(uniqueLabelInd)

                        tempEvents = tempEvents.deleteAllEventsWithThisLabel(obj.event.uniqueLabel{uniqueLabelInd});

                    end

                end

                obj.event = tempEvents;
            else
                disp('No event markers present that could be deleted.')
            end
            
        end
        
        
        %%
        function createEvents(obj,varargin)
            % Creates event markers from the data.
            %
            % markers are created based on the velocity data stream (1st child of euler
            % angle values). entered values are:
            % - channel that should be used for marker generation (4 are yaw values in our
            % case)
            % - algorithm that should be used for marker creation: 'movement', 'maxima', 'minima', 'zero crossing'
            % - one string with onset and offset marker names separated by a single
            % whitespace
            % - window length for determining the fine movement onset/offset
            % threshold (in s after coarse onset detection). set this to a value that you expect the average
            % movement to be long
            % - boolean only used in algorithms that are NOT 'movement' (if max/min criteria should only be fulfilled if
            % sign is pos/neg).
            % - VERY IMPORTANT! threshold for generally detecting a movement (values that are above are
            % classified as movement). the entered value is the percentage of datapoints
            % that are lower than the threshold (example: the threshold is set such that 65% of the
            % data are lower, meaning in practice that we assume the participant is
            % moving 35% of the time.)
            % - IMPORTANT! threshold for the fine tuned onsets and offsets in
            % percentage of the maximum value present in the window after onset detected
            % - minimum movement duration (if 'movement' is chosen as algorithm)
            % necessary for a movement to also be marked as such (in ms). the first
            % time we do this with 0, as written at the top of this loop, so all movements are detected
            % and then the appropriate minDuration is chosen.
            %
            % Input arguments:
            %       none
            %
            % Output argument:
            %       none. The events will be saved in the present data set.
            %
            % Uses:
            %       mocapObj = mobilab.allStreams.item{ mocapItem };
            %       mocapObj.createEvents(channel,method,eventName(s),windowLength,correctSign,movementThreshold,movementThresholdFine,minMovementDuration);
            
            dispCommand = false;
            if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
                prefObj = [...
                    PropertyGridField('channel',4,'DisplayName','Channel','Category','Main','Description','Channel number. Enter desired channel to generate marker from and hit return.')...
                    PropertyGridField('criteria','movements','Type',PropertyType('char', 'row', {'maxima', 'minima','zero crossing', 'sliding window deviation', 'movements'}),'DisplayName','Criteria','Category','Main','Description','Criterion for making the event, could be: maxima, minima, zero crossing, sliding window deviation, movements.')...
                    PropertyGridField('correctSign',0,'DisplayName','Correct sign criteria','Category','Main','Description','Enter if max/min criteria should only be fulfilled if sign is pos/neg and hit return.')...
                    PropertyGridField('eventType','movement:start movement:end','DisplayName','Marker name','Category','Main','Description','Enter the name of the new event marker and hit return.')...
                    PropertyGridField('inhibitionWindow',2,'DisplayName','Inhibition/sliding window length','Category','Main','Description','Enter the length of the inhibition window and hit return. Is multiplied by sampling rate!')...
                    PropertyGridField('movementThreshold',65,'DisplayName','Threshold for detecting a general movement','Category','Main','Description','Percentage of values that should be lower than the threshold. Enter the threshold and hit return.')...
                    PropertyGridField('movementOnsetThresholdFine',5,'DisplayName','Threshold for detecting a movement onset in the velocity in percentage once a movement has been detected','Category','Main','Description','Enter the threshold and hit return.')...
                    PropertyGridField('minimumDuration',0,'DisplayName','Minimum duration for a movement','Category','Main','Description','If ''movement'' category is chosen, only movements longer than this duration (in ms) are considered. Enter and hit return.')...
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
                varargin{4} = val.inhibitionWindow;
                varargin{5} = val.correctSign;
                varargin{6} = val.movementThreshold;
                varargin{7} = val.movementOnsetThresholdFine;
                varargin{8} = val.minimumDuration;
                dispCommand = true;
            end
            
            Narg = length(varargin);
            if Narg < 1, channel   = 1;       else channel   = varargin{1}(1);end
            if Narg < 2, criteria  = 'movements'; else criteria  = varargin{2};   end
            if Narg < 3, eventType = {'default'};else eventType = strsplit(varargin{3}, ' ');   end
            if Narg < 4
                inhibitedWindowLength = obj.samplingRate;
            else
                inhibitedWindowLength = ceil(obj.samplingRate*varargin{4});
            end
            if Narg < 5, correctSign = 0;   else correctSign = varargin{5}; end
            if Narg < 6, movementThreshold = 65;   else movementThreshold = varargin{6}; end
            if Narg < 7, movementOnsetThresholdFine = 0.05;   else movementOnsetThresholdFine = varargin{7} / 100; end
            if Narg < 8, minimumDuration = 0; else minimumDuration = varargin{8}; end
            if Narg < 9
                segmentObj = basicSegment([obj.timeStamp(1),obj.timeStamp(end)]);
            else
                segmentObj = varargin{8};
            end
            
            if dispCommand
                disp('Running:');
                disp('   obj.createEvents(channel,method,eventName(s),windowLength,correctSign,movementThreshold,movementThresholdFine,minMovementDuration)');
            end
            
            numberOfSegments = length(segmentObj.startLatency);
            index = obj.getTimeIndex([segmentObj.startLatency segmentObj.endLatency]);
            index = reshape(index,numberOfSegments,2);
            
            if strcmp(eventType(1),'') || strcmp(eventType{1},'default')
                switch criteria
                    case 'maxima',                      eventType = {'max'};
                    case 'zero crossing',               eventType = {'zc'};
                    case 'minima',                      eventType = {'min'};
                    case 'sliding window deviation',    eventType = {'slideDev'};
                    case 'movements',                   eventType = {'onset', 'offset'};
                    otherwise,                          eventType = {'noname'};
                end
            end
            %signal = obj.magnitude(:,channel);
            signal = obj.mmfObj.data.x(:,channel);
            
            for it=1:numberOfSegments % default is 1 segment: the whole data stream
                [I J] = searchInSegment(signal(index(it,1):index(it,2)),criteria,inhibitedWindowLength,movementThreshold, movementOnsetThresholdFine, round(obj.samplingRate*minimumDuration/1000));
                if I == 0
                    break
                end
                if correctSign && (strcmp(criteria, 'maxima') || strcmp(criteria, 'minima'))
                    if strcmp(criteria, 'minima')
                        signal = -signal;
                    end
                    I(sign(signal(I))==-1)=[];
                end
                
                time = obj.timeStamp(index(it,1):index(it,2));
                latencyI = obj.getTimeIndex(time(I));
                obj.event = obj.event.addEvent(latencyI,eventType{1});
                
                if J ~= 0
                    time = obj.timeStamp(index(it,1):index(it,2));
                    latencyJ = obj.getTimeIndex(time(J));
                    obj.event = obj.event.addEvent(latencyJ,eventType{2});
                end
                
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
                    channelsToThrow = commandHistory.varargin{2};
                    metadata.numberOfChannels = length(channelsToThrow);
                    metadata.label = obj.label(channelsToThrow);
                    metadata.artifactMask = obj.artifactMask(:,channelsToThrow);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'throwOutChannels'
                    prename = 'throwOut_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    channelsToThrow = commandHistory.varargin{1};
                    channelsToKeep = 1:obj.numberOfChannels;
                    channelsToKeep(channelsToThrow) = [];
                    
                    metadata.numberOfChannels = length(channelsToKeep);
                    metadata.label = obj.label(channelsToKeep);
                    metadata.artifactMask = obj.artifactMask(:,channelsToKeep);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(channelsToKeep)]);
                    
                case 'addChannels'
                    prename = 'addChan_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    
                    
                    allLabels = obj.label;
                    allLabels(end+1:end+length(commandHistory.varargin{1})) = commandHistory.varargin{1};
                    
                    oldArtifactMask = obj.artifactMask;
                    newArtifactMask = sparse(size(obj.data,1),length(commandHistory.varargin{1}));
                    allArtifactMask = [oldArtifactMask newArtifactMask];
                    
                    metadata.numberOfChannels = length(allLabels);
                    metadata.label = allLabels;
                    metadata.artifactMask = allArtifactMask;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(allLabels)]);
                    
                case 'degToRad'
                    prename = 'deg2rad_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    channelsToThrow = commandHistory.varargin{2};
                    metadata.numberOfChannels = length(channelsToThrow);
                    metadata.label = obj.label(channelsToThrow);
                    metadata.artifactMask = obj.artifactMask(:,channelsToThrow);
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(channelsToThrow)]);
                    
                case 'oneEuroFilter'
                    prename = 'oneEuroFilter_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = obj.numberOfChannels;
                    metadata.label = obj.label;
                    metadata.artifactMask = obj.artifactMask;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'switchCoordinateSystem'
                    prename = 'switchCoordSys_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = obj.numberOfChannels;
                    metadata.label = obj.label;
                    metadata.artifactMask = obj.artifactMask;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'unflipSigns'
                    prename = 'unflip_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = obj.numberOfChannels;
                    metadata.label = obj.label;
                    metadata.artifactMask = obj.artifactMask;
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'quaternionsToEuler'
                    prename = 'quat2eul_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
                    metadata.numberOfChannels = length(commandHistory.newLabels);
                    metadata.label = commandHistory.newLabels;
                    metadata.artifactMask = obj.artifactMask(:,1:length(commandHistory.newLabels));
                    allocateFile(metadata.binFile,metadata.precision,[length(metadata.timeStamp) length(commandHistory.newLabels)]);
                    
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
                        metadata.artifactMask = obj.artifactMask(:,channels);
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
                    metadata.artifactMask = sparse(length(metadata.timeStamp),metadata.numberOfChannels);
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
                    metadata.artifactMask = sparse(length(metadata.timeStamp),metadata.numberOfChannels);
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
                    channelsToThrow = [];
                    for it=1:length(tmp)
                        channelsToThrow(end+1) = tmp(it)-2;%#ok
                        channelsToThrow(end+1) = tmp(it)-1;%#ok
                        channelsToThrow(end+1) = tmp(it);%#ok
                    end
                    
                    fid = fopen(metadata.binFile,'w');
                    if fid<=0, error('Invalid file identifier. Cannot create the copy object.');end;
                    fwrite(data(:),obj.precision);
                    fclose(fid);
                    metadata.artifactMask = obj.artifactMask(:,channelsToThrow);
                    N = length(channelsToThrow);
                    metadata.label = cell(N,1);
                    for it=1:N, metadata.label{it} = obj.label{channelsToThrow(it)};end
                    
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
            
            menuItem = javax.swing.JMenuItem('Throw out channels');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'throwOutChannels',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Unflip signs of jumping quaternion channels');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'unflipSigns',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Filling-in occluded time points');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'removeOcclusionArtifact',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Switch between left- and right hand sided coordinate system');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'switchCoordinateSystem',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Lowpass filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'lowpass',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('One Euro filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'oneEuroFilter',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Transform Quaternions to Euler Angles');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'quaternionsToEuler',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Convert from degree to radian');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'degToRad',-1});
            jmenu.add(menuItem);
            
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Compute time derivatives');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'timeDerivative',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Create event marker');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'createEvents',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Add events from time derivative streams');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'addEventsFromDerivatives',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Delete Event Markers');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'deleteMarkers',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Time frequency analysis (CWT)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'continuousWaveletTransform',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Add stick figure');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'loadConnectedBody',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot stick figure');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'mocapBrowser',-1});
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
