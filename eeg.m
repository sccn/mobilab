% Definition of the class eeg. This class defines analysis methods
% exclusively for EEG data.
%
% For more details visit: https://code.google.com/p/mobilab/ 
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Apr-2011

classdef eeg < dataStream & headModel
    properties
        isReferenced    % Boolean that reflects whether an EEG data set has been re-referenced of not.
        reference       % Cell array with the labels of the channels used to compute the reference.
    end
    methods
        function obj = eeg(header)
            % Creates an eeg object.
            % 
            % Input arguments:
            %       header: header file (string)
            %       
            % Output arguments:
            %       obj:    eeg object (handle)
            % 
            % Usage:
            %       obj = eeg(header);

            if nargin < 1, error('Not enough input arguments.');end
            warning off %#ok
            load(header,'-mat','label','channelSpace','surfaces','atlas','fiducials','leadFieldFile'); 
            warning on  %#ok
            if ~exist('channelSpace','var'),  channelSpace  = [];save(header,'-mat','-append','channelSpace');end
            if ~exist('surfaces','var'),      surfaces      = [];save(header,'-mat','-append','surfaces');end
            if ~exist('atlas','var'),         atlas         = [];save(header,'-mat','-append','atlas');end
            if ~exist('fiducials','var'),     fiducials     = [];save(header,'-mat','-append','fiducials');end
            if ~exist('leadFieldFile','var'), leadFieldFile = [];save(header,'-mat','-append','leadFieldFile');end
            if ~isempty(surfaces)
                path = fileparts(header);
                [~,name,ext] = fileparts(surfaces);
                surfaces = fullfile(path,[name,ext]);
            end
            if ~isempty(leadFieldFile)
                path = fileparts(header);
                [~,name,ext] = fileparts(leadFieldFile);
                leadFieldFile = fullfile(path,[name,ext]);
            end
            obj@dataStream(header);
            obj@headModel('channelSpace',channelSpace,'label',label,'surfaces',surfaces,...
                'atlas',atlas,'fiducials',fiducials,'leadFieldFile',leadFieldFile);
        end
        %%
        function delete(obj)
            delete@headModel(obj);
            delete@dataStream(obj);
        end
        %%
        function isReferenced = get.isReferenced(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.set.isReferenced'))
                isReferenced = obj.isReferenced;
                return;
            end
            if isempty(obj.isReferenced)
                try obj.isReferenced = retrieveProperty(obj,'isReferenced');
                catch
                    isReferenced = false;
                    save(obj.header,'-mat','-append','isReferenced')
                    obj.isReferenced = isReferenced;
                end
            end
            isReferenced = obj.isReferenced;
        end
        function set.isReferenced(obj,isReferenced)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.get.isReferenced'))
                obj.isReferenced = isReferenced;
                return;
            end
            obj.isReferenced = isReferenced;
            saveProperty(obj,'isReferenced',isReferenced)
        end
        %%
        function reference = get.reference(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.set.reference'))
                reference = obj.reference;
                return;
            end
            if isempty(obj.reference), obj.reference = retrieveProperty(obj,'reference');end
            reference = obj.reference;
        end
        function set.reference(obj,reference)
            stack = dbstack;
            if any(strcmp({stack.name},'eeg.get.reference'))
                obj.reference = reference;
                return;
            end
            obj.reference = reference;
            saveProperty(obj,'reference',reference);
        end
        %%
        function jsonObj = serialize(obj)
            metadata = saveobj(obj);
            metadata.size = size(obj);
            metadata.event = obj.event.uniqueLabel;
            metadata.artifactMask = sum(metadata.artifactMask(:) ~= 0);
            metadata.writable = double(metadata.writable);
            metadata.history = obj.history;
            metadata.sessionUUID = char(obj.sessionUUID);
            metadata.uuid = char(obj.uuid);
            if ~isempty(metadata.channelSpace),  metadata.hasChannelSpace = 'yes'; else metadata.hasChannelSpace = 'no';end
            if ~isempty(metadata.leadFieldFile), metadata.hasLeadField    = 'yes'; else metadata.hasLeadField    = 'no';end
            if ~isempty(obj.fiducials) &&  ~isempty(obj.surfaces) && ~isempty(obj.atlas)
                 metadata.hasHeadModel = 'yes';
            else metadata.hasHeadModel = 'no';
            end
            metadata = rmfield(metadata,{'parentCommand' 'timeStamp' 'hardwareMetaData' 'channelSpace' 'leadFieldFile' 'fiducials' 'surfaces' 'atlas'});
            jsonObj = savejson('',metadata,'ForceRootName', false);
        end
        %%
        function buildHeadModelFromTemplate(obj,headModelFile,showModel)
            % Builds an individualized head model that match the sensor position of the subject.
            % This methodology represents an alternative to cases where the MRI of the subject 
            % is not available.
            %
            % Input arguments:
            %       headModelFile: pointer to the template head model
            %       showModel:     if true a figure with the resulting 
            %                      individualized head model is showed
            %                         
            % Usage:
            %       eegObj        = mobilab.allStreams.item{ eegItem };
            %       headModelFile = mobilab.preferences.eeg.headModel;
            %       showModel     = true;
            %       eegObj.buildHeadModelFromTemplate( headModelFile, showModel);
            
            
            if isempty(obj.channelSpace) || isempty(obj.label)
                error('Cannot build a head model without channel locations or fiducials.');
            end
            if nargin < 2,
                [filename, pathname] = uigetfile2('*.mat','Select the mat file containing 3 surfaces (scalp, skull, brain)',...
                    obj.container.container.preferences.eeg.headModel);
                if ~all([ischar(filename) ischar(pathname)]), return;end
                headModelFile = fullfile(pathname,filename);
            end
            if nargin < 3, showModel = true;end
            if ~ischar(headModelFile)
                [filename, pathname] = uigetfile2('*.mat','Select the mat file containing at least 3 surfaces (1-scalp, 2-skull, 3-brain)',...
                    obj.container.container.preferences.eeg.headModel);
                if ~all([ischar(filename) ischar(pathname)]), return;end
                headModelFile = fullfile(pathname,filename);
                disp('Running:');
                disp(['  mobilab.allStreams.item{' num2str(obj.container.findItem(obj.uuid)) '}.buildHeadModelFromTemplate(''' headModelFile ''', plotModelFlag );']);
            end
            if ~exist(headModelFile,'file'), error('MoBILAB:missingAnatomicalModel',...
                    'Missing Standardized head model. Check ''default head model'' in MoBILAB''s preferences.');
            end
            
            individualHeadModel = fullfile(obj.container.mobiDataDirectory,['headModel_' obj.name '_' obj.uuid '_' obj.sessionUUID '.mat']);
            obj.surfaces = obj.warpTemplate2channelSpace(headModelFile,individualHeadModel);
            saveProperty(obj,'surfaces',obj.surfaces);
            saveProperty(obj,'atlas',obj.atlas);
            if showModel, obj.plotHeadModel;end
        end
        %%
        function readMontage(obj,file)
            % Reads a file containing the sensor positions and its labels. 
            % If recorder, it also extracts fiducial landmarks. If the number
            % of channels in the file are less than the number of channels in
            % the object, a new object is created with the common set. If more
            % than one file is supplied the program creates separate objects 
            % containing the different set of channels selected by each file.
            % 
            % Input argument: 
            %       file: filename. The following formats are supported: 
            %             BESA or EGI3D cartesian .sfp, Polhemus .elp, 
            %             Matlab .xyz or .sph, EEGLAB .loc, and Neuroscan
            %             .asc or .dat.

            if nargin < 2
                [filename, pathname] = uigetfile2({'*.sfp','BESA or EGI 3-D cartesian files (*.sfp)';'*.elp','Polhemus native files (*.elp)';...
                    '*.xyz','Matlab xyz files (*.xyz)';'*.loc','EEGLAB polar files (*.loc)';'*.sph','Matlab spherical files (*.sph)';...
                    '*.asc','Neuroscan polar files (*.asc)';'*.dat','Neuroscan 3-D files (*.dat)';'*.*','All files (*.*)'},...
                    'Select a file containing sensor positions','MultiSelect', 'on');
                if isnumeric(filename) || isnumeric(pathname), return;end
            elseif ~ischar(file)
                [filename, pathname] = uigetfile2({'*.sfp','BESA or EGI 3-D cartesian files (*.sfp)';'*.elp','Polhemus native files (*.elp)';...
                    '*.xyz','Matlab xyz files (*.xyz)';'*.loc','EEGLAB polar files (*.loc)';'*.sph','Matlab spherical files (*.sph)';...
                    '*.asc','Neuroscan polar files (*.asc)';'*.dat','Neuroscan 3-D files (*.dat)';'*.*','All files (*.*)'},...
                    'Select a file containing sensor positions','MultiSelect', 'on');
                if isnumeric(filename) || isnumeric(pathname), return;end
            else
                [pathname,filename,ext] = fileparts(file);
                filename = [filename,ext];
            end
            
            if iscellstr(filename)
                N = length(filename);
                newChannelSpace = cell(N,1);
                newLabels = cell(N,1);
                newEEGchannel = cell(N,1);
                fiducials = cell(N,1);
                
                tmpLabel = [];
                indEEGchannels = false(obj.numberOfChannels,1);
                for it=1:obj.numberOfChannels
                    ind = obj.label{it} == '_';
                    if any(ind)
                        ind = find(ind);
                        tmpLabel{end+1} = obj.label{it}(ind(end)+1:end); %#ok
                        indEEGchannels(it) = true;
                    end
                end
                indEEGchannels = find(indEEGchannels);
                if isempty(indEEGchannels)
                    indEEGchannels = 1:obj.numberOfChannels;
                    tmpLabel = obj.label;
                end
                for it=1:N
                    file = fullfile(pathname,filename{it});
                    [eloc, labels] = readlocs( file);
                    eloc = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
                    Nl = length(labels);
                    elecIndices = false(Nl,1);
                    channelIndices = zeros(Nl,1);
                    for jt=1:Nl
                        if ~isempty(strfind(labels{jt},'fidnz')) || ~isempty(strfind(labels{jt},'nasion'))
                            fiducials{it}.nasion = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt9')) || ~isempty(strfind(labels{jt},'lpa'))
                            fiducials{it}.lpa = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt10')) || ~isempty(strfind(labels{jt},'rpa'))
                            fiducials{it}.rpa = eloc(jt,:);
                        elseif ~isempty(strfind(labels{jt},'fidt10')) || ~isempty(strfind(labels{jt},'vertex'))
                            fiducials{it}.vertex = eloc(jt,:);    
                        else
                            I = find(strcmpi(tmpLabel,labels{jt}));
                            if ~isempty(I)
                                elecIndices(jt) = true;
                                channelIndices(jt) = I(1);
                            end
                        end
                    end
                    channelIndices(channelIndices==0) = [];
                    
                    newLabels{it} = labels(elecIndices);
                    newChannelSpace{it} = eloc(logical(elecIndices),:);
                    newEEGchannel{it} = indEEGchannels(channelIndices);
                    tmpLabel(channelIndices) = []; %#ok
                    indEEGchannels(channelIndices) = [];
                end
                % these two loops can be fused, however I've split them for debugging
                for it=1:N
                    cobj = divideStreamObject(obj,newEEGchannel{it},newLabels{it},[obj.name '_' num2str(it)]);
                    cobj.channelSpace = newChannelSpace{it};
                    saveProperty(cobj,'channelSpace',cobj.channelSpace);
                    try
                        cobj.fiducials = fiducials{it};
                        saveProperty(cobj,'fiducials',cobj.fiducials);
                    catch ME
                        warning(ME.message);
                        warning('MoBILAB:noFiducials','Fiducial landmarks are missing.');
                    end
                end
            else
                file = fullfile(pathname,filename);
                [eloc, labels, theta, radius, indices] = readlocs( file); %#ok
                eloc = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
                rmThis = [];
                I = strcmpi(labels,'fidnz') | strcmpi(labels,'nasion'); 
                if any(I), fiducials.nasion = eloc(I,:);rmThis(1) = find(I);end
                
                I = strcmpi(labels,'fidt9') | strcmpi(labels,'lpa');     
                if any(I), fiducials.lpa = eloc(I,:);rmThis(2) = find(I);end
                
                I = strcmpi(labels,'fidt10') | strcmpi(labels,'rpa');    
                if any(I), fiducials.rpa = eloc(I,:);rmThis(3) = find(I);end
                
                I = strcmpi(labels,'vertex');
                if any(I), fiducials.vertex = eloc(I,:);rmThis(4) = find(I);end
                
                % fixing sccn old protocol bug
                for it=1:length(labels), if ~isempty(strfind(labels{it},'EXG')), labels{it}(strfind(labels{it},'EXG'):3) = 'EXT';end;end
                
                labels(rmThis) = [];
                eloc(rmThis,:) = [];
                
                if obj.numberOfChannels > length(labels) 
                    ind = obj.container.getItemIndexFromItemNameSimilarTo(obj.name);
                    tmpObj = divideStreamObject(obj,1:length(labels) ,labels,[obj.name '_' num2str(length(ind))]);
                else
                    tmpObj = obj;
                end
                tmpObj.label = labels;
                tmpObj.channelSpace = eloc;
                saveProperty(tmpObj,'channelSpace',tmpObj.channelSpace);
                if exist('fiducials','var')
                    tmpObj.fiducials = fiducials;
                    saveProperty(tmpObj,'fiducials',tmpObj.fiducials);
                else
                    warning('MoBILAB:noFiducials','Fiducial landmarks are missing.');
                end
            end
            disp('Done!!!')
        end
        %%
        function browserObj = plotOnScalp(obj,defaults)
            % Plot EEG data on the surface of the scalp.
            
            if isempty(obj.channelSpace) || isempty(obj.label) || isempty(obj.surfaces);
                error('Head model is incomplete or missing.');
            end
            if nargin < 2, defaults.browser  = @topographyBrowserHandle;end
            if isnumeric(defaults), clear defaults;defaults.browser  = @topographyBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser  = @topographyBrowserHandle;end
            browserObj = topographyBrowserHandle(obj,defaults);
        end
        %%
        function cobj = filter(obj,varargin)
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            cobj = filter@dataStream(obj,varargin);
            if isempty(cobj), return;end
            cobj.mmfObj.Writable = true;
            obj.initStatusbar(0,1,'Centering channels...');
            cobj.mmfObj.Data.x = bsxfun(@minus,cobj.mmfObj.Data.x,mean(cobj.mmfObj.Data.x));
            obj.statusbar(1);
            cobj.mmfObj.Writable = false;
        end
        %%
        function cobj = removeSubspace(obj,noise)
            dispCommand = false;
            if nargin < 2, noise = obj.auxChannel.data;end
            if all(noise(:) == -1)
                noise = obj.auxChannel.data;
                dispCommand = true;
            end
            if isempty(noise)
                disp('auxChannel is empty.')
                return;
            end
            
            try
                commandHistory.commandName = 'removeSubspace';
                commandHistory.uuid = obj.uuid;
                cobj = obj.copyobj(commandHistory);
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                pval   = 0.01;
                aBand  = [7 13];
                hp     = obj.firDesign(obj.samplingRate*4,'highpass',1);
                bs     = obj.firDesign(obj.samplingRate*4,'bandstop',aBand);
                noise  = filtfilt_fast(hp,1,noise);
                noise  = filtfilt_fast(bs,1,noise);
                data   = obj.mmfObj.Data.x;
                delta  = fix(obj.samplingRate*2);
                dim    = size(obj);
                clThis = false(dim(2),1);
                warning off all
                obj.initStatusbar(1,dim(2),'Removing subspace...');
                tic;
                fprintf('Now cleaning:');
                for ch=1:dim(2)
                    fprintf(' %s(%i)',obj.label{ch},ch);
                    y     = data(:,ch);
                    hType = hann(delta*2);
                    for it=delta:delta:dim(1)
                        if it+delta < dim(1), ind = it-delta+1:it+delta; else ind = it-delta+1:dim(1);hType=hann(length(ind));end
                        [~,stats] = robustfit(noise(ind,:),data(ind,ch));
                        significantCoef = stats.p(1:end-1) < pval;
                        if any(significantCoef)
                            %[~,stats] = robustfit(noise(ind,significantCoef),data(ind,ch));
                            %if any(stats.p(1:end-1) < pval)
                               % y(ind) = ( [y(ind(1:delta)) y(ind(delta+1:end))].*hType + stats.resid)/2; 
                               y(ind) = (y(ind).*hType + stats.resid)/2;
                            %end
                        end
                    end
                    if any(y~=data(:,ch)),
                        cobj.mmfObj.Data.x(:,ch) = y;
                        clThis(ch) = true;
                    end
                    obj.statusbar(ch);
                end
                toc
            catch ME
                obj.statusbar(inf);
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
            warning on; %#ok
        end
        %%
        function cobj = reReference(obj,channels2BeReferenced,channels2BeAveraged)
            % Re-reference the data.
            % 
            % Input arguments:
            %       channels2BeReferenced: cell array with the label of the
            %                              channels to be referenced
            %       channels2BeAveraged:   cell array with the label of the
            %                              channels to be averaged
            %          
            % Output arguments:
            %       cobj: handle to the new object
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       channels2BeReferenced = eegObj.label;
            %       channels2BeAveraged   = eegObj.label;
            %       eegObjRef = eegObj.reReference(channels2BeReferenced,channels2BeAveraged);
            
            if nargin < 2, error('Not enough input arguments.');end
            dispCommand = false;
            if isnumeric(channels2BeReferenced) && channels2BeReferenced==-1
                cobj = [];
                h = ReReferencing(obj);
                uiwait(h);
                try userData = get(h,'userData');catch return;end %#ok
                close(h);
                drawnow
                if ~iscell(userData), return;end
                channels2BeReferenced = userData{1};
                channels2BeAveraged = userData{2};
                dispCommand = true;
            end
            if ~iscellstr(channels2BeReferenced), error('First argument must be a cell with the labels for the channels to be referenced.');end
            if ~iscell(channels2BeAveraged), error('Second argument must be a cell with the labels for the channels to be averaged (The ones to be taken as a reference).');end
            
            try
                ind1 = ismember(obj.label,channels2BeReferenced);
                ind2 = ismember(obj.label,channels2BeAveraged);
                
                if sum(ind1)*sum(ind2)==0, error('Some of the labels you''ve entered don''t match the labels in the object.');end
                
                ind1 = sort(find(ind1));
                ind2 = sort(find(ind2));
                
                commandHistory.commandName = 'reReference';
                commandHistory.uuid = obj.uuid;
                commandHistory.varargin{1} = obj.label(ind1);
                commandHistory.varargin{2} = obj.label(ind2);
                
                %I = ismember(ind1,ind2);
                %if any(I), ind1(end) = [];end
                
                cobj = obj.copyobj(commandHistory);
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                data = obj.mmfObj.Data.x;
                
                ref = mean(data(:,ind2),2);
                % cobj.mmfObj.Data.x = bsxfun(@minus,data(:,ind1),ref);
                obj.initStatusbar(1,cobj.numberOfChannels,'Re-referencing...');
                for it=1:cobj.numberOfChannels
                    cobj.mmfObj.Data.x(:,it) = data(:,ind1(it))-ref;
                    obj.statusbar(it);
                end
                cobj.isReferenced = true;
                cobj.reference = channels2BeAveraged;
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function cobj = artifactsRejection(obj,varargin)
            % This method uses Christian Kothe's clean_artifacts toolbox
            
            dispCommand = false;
            if  ~isempty(varargin) && length(varargin{1}) == 1 && isnumeric(varargin{1}) && varargin{1} == -1
                prefObj = [...
                    PropertyGridField('channel_crit',0.45,'DisplayName','ChannelCriterion','Category','Main','Description','Criterion for removing bad channels. This is a minimum correlation value that a given channel must have w.r.t. at least one other channel. Generally, a fraction of most correlated channels is excluded from this measure. A higher value makes the criterion more aggressive. Reasonable range: 0.45 (very lax) - 0.65 (quite aggressive).')...
                    PropertyGridField('burst_crit',3,'DisplayName','BurstCriterion','Category','Main','Description','Criterion for projecting local bursts out of the data. This is the standard deviation from clean EEG at which a signal component would be considered a burst artifact. Generally a higher value makes the criterion less aggressive. Reasonable range: 2.5 (very aggressive) to 5 (very lax). One usually does not need to tune this parameter.')...
                    PropertyGridField('window_crit',0.1,'DisplayName','WindowCriterion','Category','Main','Description','Criterion for removing bad time windows. This is a quantile of the per-window variance that should be considered for removal. Multiple channels need to be in that quantile for the window to be removed. Generally a higher value makes the criterion more aggressive. Reasonable range: 0.05 (very lax) to 0.15 (quite aggressive).')...
                    PropertyGridField('highpass',[0.5 1],'DisplayName','Highpass','Category','Main','Description','Transition band for the initial high-pass filter in Hz. This is [transition-start, transition-end].')...
                    PropertyGridField('interp_crit_channel',true,'DisplayName','InterpCrtiticalChannel','Category','Main','Description','If true, interpolates critical channels using local-linear spacial Gaussian kernel. It uses the coordinates of three neighbor sensors to reconstruct the signal of the hopeless channels.')...
                    PropertyGridField('channel_crit_excluded',0.1,'DisplayName','ChannelCriterionExcluded','Category', 'Specialty','Description','The fraction of excluded most correlated channels when computing the Channel criterion. This adds robustness against channels that are disconnected and record the same noise process. At most this fraction of all channels may be fully disconnected. Reasonable range: 0.1 (fairly lax) to 0.3 (very aggressive); note that increasing this value requires the ChannelCriterion to be relaxed to maintain the same overall amount of removed channels.')...
                    PropertyGridField('channel_crit_maxbad_time',0.5,'DisplayName','ChannelCriterionMaxBadTime','Category', 'Specialty','Description','This is the maximum fraction of the data set during which a channel may be bad without being considered "bad". Generally a lower value makes the criterion more aggresive. Reasonable range: 0.15 (very aggressive) to 0.5 (very lax).')...
                    PropertyGridField('burst_crit_refmaxbadchns',0.075,'DisplayName','BurstCriterionRefMaxBadChns','Category', 'Specialty','Description','The maximum fraction of bad channels per time window of the data that is used as clean reference EEG for the burst criterion. Instead of a number one may also directly pass in a data set that contains clean reference data (for example a minute of resting EEG; this has to be done from the command line). Reasonable range: 0.05 (very aggressive) to 0.3 (quite lax).')...
                    PropertyGridField('burst_crit_reftolerances',[-5 3.5],'DisplayName','BurstCriterionRefTolerances','Category', 'Specialty','Description','These are the power tolerances beyond which a channel in the clean reference data is considered "bad", in standard deviations relative to a robust EEG power distribution (lower and upper bound).')...
                    PropertyGridField('window_crit_tolerances',[-5 5],'DisplayName','WindowCriterionTolerances','Category', 'Specialty','Description','These are the power tolerances beyond which a channel in the final output data is considered "bad", in standard deviations relative to a robust EEG power distribution (lower and upper bound).')...
                    PropertyGridField('flatline_crit',5,'DisplayName','FlatlineCriterion','Category', 'Specialty','Description','Maximum tolerated flatline duration. In seconds. If a channel has a longer flatline than this, it will be considered abnormal. Default: 5')...
                    ];

                hFigure = figure('MenuBar','none','Name','EEG artifacts rejection','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
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
                varargin{1} = val.channel_crit;
                varargin{2} = val.burst_crit;
                varargin{3} = val.window_crit;
                varargin{4} = val.highpass;
                varargin{5} = val.interp_crit_channel;
                varargin{6} = val.channel_crit_excluded;
                varargin{7} = val.channel_crit_maxbad_time;
                varargin{8} = val.burst_crit_refmaxbadchns;
                varargin{9} = val.burst_crit_reftolerances;
                varargin{10} = val.window_crit_tolerances;
                varargin{11} = val.flatline_crit;
                dispCommand = true;
            end
            
            Narg = length(varargin);
            if Narg < 1, channel_crit = 0.45; else channel_crit = varargin{1};end
            if Narg < 2, burst_crit = 3; else burst_crit = varargin{2};end
            if Narg < 3, window_crit = 0.1; else window_crit = varargin{3};end
            if Narg < 4, highpass = [0.5 1]; else highpass = varargin{4};end
            if Narg < 5, interp_crit_channel = true; else interp_crit_channel = varargin{5};end
            if Narg < 6, channel_crit_excluded = 0.1; else channel_crit_excluded = varargin{6};end
            if Narg < 7, channel_crit_maxbad_time = 0.5; else channel_crit_maxbad_time = varargin{7};end
            if Narg < 8, burst_crit_refmaxbadchns = 0.075; else burst_crit_refmaxbadchns = varargin{8};end
            if Narg < 9, burst_crit_reftolerances = [-5 3.5]; else burst_crit_reftolerances = varargin{9};end
            if Narg < 10, window_crit_tolerances = [-5 5]; else window_crit_tolerances = varargin{10};end
            if Narg < 11, flatline_crit = 5; else flatline_crit = varargin{11};end
                        
            try
                
                commandHistory.commandName = 'artifactsRejection';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = channel_crit;
                commandHistory.varargin{2} = burst_crit;
                commandHistory.varargin{3} = window_crit;
                commandHistory.varargin{4} = highpass;
                commandHistory.varargin{5} = interp_crit_channel;
                commandHistory.varargin{6} = channel_crit_excluded;
                commandHistory.varargin{7} = channel_crit_maxbad_time;
                commandHistory.varargin{8} = burst_crit_refmaxbadchns;
                commandHistory.varargin{9} = burst_crit_reftolerances;
                commandHistory.varargin{10} = window_crit_tolerances;
                commandHistory.varargin{11} = flatline_crit;
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                EEG = cobj.EEGstructure;
                
                existCleaningCombo = exist('clean_artifacts','file');
                if ~existCleaningCombo, addpath(genpath([obj.container.container.path filesep 'dependency' filesep 'clean_rawdata']));end
                EEG = clean_artifacts(EEG,channel_crit,burst_crit,window_crit,highpass,channel_crit_excluded,channel_crit_maxbad_time,...
                    burst_crit_refmaxbadchns,burst_crit_reftolerances,window_crit_tolerances,flatline_crit);
                if ~existCleaningCombo, rmpath(genpath([obj.container.container.path filesep 'dependency' filesep 'clean_rawdata']));end
                
                cobj.mmfObj.Data.x(EEG.etc.clean_sample_mask,EEG.etc.clean_channel_mask) = EEG.data';
                cobj.artifactMask(~EEG.etc.clean_sample_mask,:) = 1;
                
                if interp_crit_channel
                    if sum(EEG.etc.clean_channel_mask) < obj.numberOfChannels && 100*sum(~EEG.etc.clean_channel_mask)/obj.numberOfChannels < 20
                        gTools = geometricTools;
                        W = gTools.localGaussianInterpolator(obj.channelSpace(EEG.etc.clean_channel_mask,:),obj.channelSpace(~EEG.etc.clean_channel_mask,:),3);
                        cobj.mmfObj.Data.x(EEG.etc.clean_sample_mask,~EEG.etc.clean_channel_mask) = (W*double(EEG.data))';
                    elseif 100*sum(~EEG.etc.clean_channel_mask)/obj.numberOfChannels >= 20
                        data = cobj.mmfObj.Data.x(:,EEG.etc.clean_channel_mask);
                        cobj.numberOfChannels = sum(EEG.etc.clean_channel_mask);
                        cobj.disconnect;
                        fid = fopen(cobj.binFile,'w');
                        fwrite(fid,data(:),cobj.precision);
                        fclose(fid);
                        cobj.connect;
                        cobj.artifactMask(:,~EEG.etc.clean_channel_mask) = [];
                        cobj.disconnect;
                    end
                end
                %-- Applying soft probability mask
                clear EEG
                latency = cobj.event.getLatencyForEventLabel('boundary')';
                if ~isempty(latency) && ~all(cobj.artifactMask(latency,1)), cobj.artifactMask(latency,:) = 1;end
                artProbMask = cobj.artifactMask(:,1);
                [cobj.mmfObj.Data.x,artProbMask] = softMasking(cobj.mmfObj.Data.x,artProbMask,0.5*cobj.samplingRate);
                cobj.artifactMask = sparse(repmat(artProbMask,1,obj.numberOfChannels));
                %--
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(cobj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function EEG = EEGstructure(obj,ismmf, passData)
            % Returns the equivalent EEG (EEGLAB) structure.
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       EEG    = eegObj.EEGstructure;
            
            if nargin < 2, ismmf = false;end
            if nargin < 3, passData = true;end
            s = dbstack;
            
            isCalledFromGui = any(~cellfun(@isempty,strfind({s.name},'myDispatch')));
            isCalledFromIca = any(~cellfun(@isempty,strfind({s.name},'icaEEG')));
            % updateEEGLAB    = sum(~cellfun(@isempty,strfind({s.name},'EEGstructure'))) == length(s);
            if isCalledFromGui
                disp('Running:');
                disp(['  EEG = mobilab.allStreams.item{ ' num2str(obj.container.findItem(obj.uuid)) ' }.EEGstructure;']);
            end
            
            EEG = eeg_emptyset;
            EEG.setname = obj.name;
            EEG.times = obj.timeStamp*1e3;     % from seconds to mili-seconds
            EEG.nbchan = obj.numberOfChannels;
            EEG.pnts = length(obj.timeStamp);
            EEG.trials = 1;
            EEG.srate = obj.samplingRate;
            EEG.xmin = obj.timeStamp(1);
            EEG.xmax = obj.timeStamp(end);
            
            [path,name,ext] = fileparts(obj.binFile); %#ok
            if passData
                if ismmf
                    EEG.data = [path filesep name '.fdt'];
                    EEG.filepath = path;
                    EEG.filename = name;
                    copyMMF(EEG,obj);
                else EEG.data = single(obj.mmfObj.Data.x)';
                end
            end
            
            EEG.chaninfo.nosedir = '+X';
            if isfield(obj.hardwareMetaData,'desc'), EEG.etc.desc = obj.hardwareMetaData.desc;end
            if ~isempty(obj.fiducials), EEG.etc.fiducials = obj.fiducials;end
                                   
            chanlocs = repmat(struct('labels',[],'type',[],'X',[],'Y',[],'Z',[],'radius',[],'theta',[]),EEG.nbchan,1);
            labels = obj.label;
            if ~isempty(obj.channelSpace)
                xyz = obj.channelSpace;
                xyz = bsxfun(@minus,xyz,mean(xyz));
                xyz = bsxfun(@rdivide,xyz,max(xyz))/2;
                for it=1:obj.numberOfChannels
                    chanlocs(it).labels = labels{it};
                    try chanlocs(it).type = obj.hardwareMetaData.desc.channels.channel{it}.type;
                    catch chanlocs(it).type = 'EEG';
                    end
                    chanlocs(it).X = xyz(it,1);
                    chanlocs(it).Y = -xyz(it,2);
                    chanlocs(it).Z = xyz(it,3);
                    [chanlocs(it).theta,chanlocs(it).radius] = cart2pol(chanlocs(it).X, chanlocs(it).Y, chanlocs(it).Z);
                    chanlocs(it).theta = chanlocs(it).theta*180/pi;
                end
                EEG.chanlocs = chanlocs;
            end
            EEG.etc.mobi.sessionUUID = obj.sessionUUID;
            
            try ALLEEG = evalin('base','ALLEEG');
            catch ALLEEG = [];%#ok
            end
                        
            if isempty(obj.event.label)
                if ismmf, pop_saveset( EEG, [name '.set'],path);end
                if ~isCalledFromIca && isCalledFromGui
                    [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);
                    assignin('base','ALLEEG',ALLEEG);
                    assignin('base','CURRENTSET',CURRENTSET);
                    assignin('base','EEG',EEG);
                    evalin('base','eeglab redraw');
                end
                return;
            end
            
            type = obj.event.label;
            latency = obj.event.latencyInFrame;
            loc = ismember(type,'boundary');
            if sum(loc)
                loc = find(loc);
                [~,loc2] = unique(latency(loc));
                [~,rmloc] = setdiff(loc,loc(loc2));
                latency(loc(rmloc)) = [];
                type(loc(rmloc)) = [];
            end
            
            if ~isempty(latency) && ~isempty(type)
                [latency,loc] = sort(latency,'ascend');
                type = type(loc); 
                EEG.event = repmat(struct('type','','latency',0,'duration',0,'urevent',1),1,length(latency));
                for it=1:length(latency)
                    EEG.event(it).type = type{it};
                    EEG.event(it).latency = latency(it);
                    EEG.event(it).urevent = it;
                    if strcmp(type,'boundary'), EEG.event(it).duration = NaN;end
                end
                EEG.urevent = EEG.event;
            end
            if ismmf && passData, pop_saveset( EEG, [name '.set'],path);end
            if ~isCalledFromIca && isCalledFromGui
                [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);
                assignin('base','ALLEEG',ALLEEG);
                assignin('base','CURRENTSET',CURRENTSET);
                assignin('base','EEG',EEG);
                evalin('base','eeglab redraw');
            end
        end
        %%
        function [J,cobj,roiObj] = estimatePCD(obj,latency,maxTol,maxIter,gridSize,verbose)
            % Computes the posterior distribution of the Primary Current Density given a
            % topographic voltage. For more details see the help section of the function
            % variationalDynLoreta.
            %
            % Input arguments:
            %       latencies: vector of latencies specifying the topographic maps to invert.
            %       maxTol:    maximum tolerance in the convergence error
            %                  of the hyperparameters, default: 1e-3
            %       maxIter:   maximum number of iterations, default: 100
            %       gridSize:  size of the grid where search for the first hyperparameters,
            %                  default: 100
            %       verbose:   if true outputs messages about the convergence of the estimation 
            %                  process, default: true
            %
            % Output arguments:
            %       J:         Primary Current Density size number of verices in the cortical
            %                  surface by number of latencies
            %       cobj:      if requested, an object is created to store the solution J 
            %       roiObj:    if requested, an object is created to store the solution J 
            %                  collapsed in anatomical regions defined by
            %                  the atlas
            % 
            % Usage:
            %       eegObj  = mobilab.allStreams.item{ eegItem };
            %       latency = 512:1024; % some latencies of interest
            %       J = eegObj.estimatePCD(latency);
            

            dispCommand = false;
            if nargin < 2, error('Specify the latency (in samples) to localize.');end
            if isempty(obj.channelSpace) || isempty(obj.label) || isempty(obj.surfaces);
                error('Head model is incomplete or missing.');
            end
            if isempty(obj.leadFieldFile), error('Lead field is missing.');end
            if isnumeric(latency) && latency(1) == -1
                 prefObj = [...
                    PropertyGridField('latency',[1 size(obj,1)],'DisplayName','Latencies','Description','Latencies (in samples) of the topographic maps to invert.')...
                    PropertyGridField('maxTol',1e-3,'DisplayName','MaxTol','Description','Maximum tolerance in the convergence error of the hyperparameters.')...
                    PropertyGridField('maxIter',100,'DisplayName','MaxIter','Description','Maximum number of iterations.')...
                    PropertyGridField('gridSize',100,'DisplayName','GridSize','Description','Size of the grid where search for the first hyperparameters.')...
                    PropertyGridField('verbose',true,'DisplayName','Verbose','Description','Outputs messages about the convergence of the estimation process.')...
                    ];

                hFigure = figure('MenuBar','none','Name','Eestimate current source','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
                    'Resize','off','userData',0);
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
                if ~get(hFigure,'userData'), close(hFigure);return;end
                close(hFigure);
                drawnow
                val = g.GetPropertyValues();
                latency = val.latency;
                if length(latency) == 2, latency = latency(1):latency(2);end
                verbose = val.verbose;
                maxTol = val.maxTol;
                maxIter = val.maxIter;
                gridSize = val.gridSize;
                
                dispCommand = true;
            end
            if ~exist('maxTol','var'),   maxTol = 1e-3;end
            if ~exist('maxIter','var'),  maxIter = 100;end
            if ~exist('gridSize','var'), gridSize = 100;end
            if ~exist('verbose','var'),  verbose = true;end
                        
            if dispCommand
                disp('Running:');
                fprintf('  mobilab.allStreams.item{%i}.estimatePCD( latency, %f, %i, %i, %i);\n',obj.container.findItem(obj.uuid),maxTol,maxIter,gridSize,verbose);
            end
            
            options.maxTol   = maxTol;
            options.maxIter  = maxIter;
            options.gridSize = gridSize;
            options.verbose  = verbose;
            
            commandHistory.commandName = 'estimatePCD';
            commandHistory.uuid = obj.uuid;
            commandHistory.varargin{1} = latency;
            commandHistory.varargin{2} = maxTol;
            commandHistory.varargin{3} = maxIter;
            commandHistory.varargin{4} = gridSize;
            commandHistory.varargin{5} = verbose;
            cobj = obj.copyobj(commandHistory);
            
            % roiObj = createROIobj(obj,commandHistory);
            % roiIndices = cell(roiObj.numberOfChannels,1);
            % colorTable = unique(obj.atlas.colorTable);
            % for it=1:length(colorTable), roiIndices{it} = obj.atlas.colorTable == colorTable(it);end
            
            
            % opening the surfaces by the Thalamus
            disp('Opening the surfaces by the Thalamus');
            structName = {'Thalamus_L' 'Thalamus_R'};
            [~,K,L,rmIndices] = getSourceSpace4PEB(obj,structName);
            load(obj.surfaces);
            n = size(surfData(end).vertices,1); %#ok
            ind = setdiff(1:n,rmIndices);
            dim = size(K);
            hasDirection = n == dim(2)/3+length(rmIndices);
            if hasDirection
                tmp = false(n,3);
                tmp(ind,:) = true;
                tmp = tmp(:);
                ind = find(tmp);
            end    
            
            % removing the average reference
            disp('Removing the average reference');
            Y = obj.data(latency,:)';
            H = eye(obj.numberOfChannels) - ones(obj.numberOfChannels)/obj.numberOfChannels;
            Y = H*Y;
            K = H*K;
            Y(end,:) = [];
            K(end,:) = [];
            Nt = length(latency);
            
            %--
            disp('Computing svd...')
            [U,S,V] = svd(K/L,'econ');
            Ut = U';
            s2 = diag(S).^2;
            iLV = L\V;
            %--
            delta = 32;
            if size(cobj,2) < 3*delta, delta=1;end
            obj.initStatusbar(1,Nt,'Estimating Primary Current Density...');
            [cobj.mmfObj.Data.x(ind,1:delta),alpha,beta] = variationalDynLoreta(Ut,Y(:,1:delta),s2,iLV,L,[],[],options);
            for it=delta+1:delta:Nt
                %J(ind,it) = inverseSolutionLoreta(Y(:,it),K,L,nlambdas,plotCSD,threshold);
                %J(ind,it) = dynamicLoreta(Ut,Y(:,it),s2,V,iLV,L);
                if it+delta <= Nt
                    [cobj.mmfObj.Data.x(ind,it-delta:it+delta-1),alpha,beta] = variationalDynLoreta(Ut,Y(:,it-delta:it+delta-1),s2,iLV,L,alpha,beta,options);
                    % for jt=1:roiObj.numberOfChannels, roiObj.mmfObj.Data.x(it-delta:it+delta-1,jt) = median(cobj.mmfObj.Data.x(roiIndices{jt},it-delta:it+delta-1))';end
                else
                    [cobj.mmfObj.Data.x(ind,it-delta:end),alpha,beta] = variationalDynLoreta(Ut,Y(:,it-delta:end),s2,iLV,L,alpha,beta,options);
                    % for jt=1:roiObj.numberOfChannels, roiObj.mmfObj.Data.x(it-delta:end,jt) = median(cobj.mmfObj.Data.x(roiIndices{jt},it-delta:end))';end
                end
                obj.statusbar(it);
            end
            obj.statusbar(Nt);
            % if plotCSD, obj.plotOnModel(J,Yp(:,1),['VariationalLoreta ' obj.name]);end
            if nargout == 1
                try J = cobj.mmfObj.Data.x;
                catch ME
                    obj.container.deleteItem(obj.container.findItem(cobj.uuid));
                    obj.container.deleteItem(obj.container.findItem(roiObj.uuid));
                    ME.rethrow;
                end
                obj.container.deleteItem(obj.container.findItem(cobj.uuid));
                obj.container.deleteItem(obj.container.findItem(roiObj.uuid));
            else J = [];
            end
        end
        %%
        function roiObj = stateSpaceGeometricAnalysis(obj,latency,maxTol,maxIter,gridSize,verbose,stateSpaceReducedDimension)
            dispCommand = false;
            if nargin < 2, error('Specify the latency (in samples) to localize.');end
            if isempty(obj.channelSpace) || isempty(obj.label) || isempty(obj.surfaces);
                error('Head model is incomplete or missing.');
            end
            if isempty(obj.leadFieldFile), error('Lead field is missing.');end
            if isnumeric(latency) && latency(1) == -1
                 prefObj = [...
                    PropertyGridField('latency',[1 size(obj,1)],'DisplayName','Latencies','Description','Latencies (in samples) of the topographic map to estimate the sources of.')...
                    PropertyGridField('maxTol',1e-3,'DisplayName','MaxTol','Description','Maximum tolarance in the convergence error of the hyperparameters.')...
                    PropertyGridField('maxIter',100,'DisplayName','MaxIter','Description','Maximum number of iterations.')...
                    PropertyGridField('gridSize',100,'DisplayName','GridSize','Description','Size of the grid where search for the first hyperparameters.')...
                    PropertyGridField('verbose',true,'DisplayName','Verbose','Description','Outputs messages about the convergence in the estimation process.')...
                    PropertyGridField('stateSpaceReducedDimension',3,'DisplayName','StateSpaceReducedDimension','Description','State space reduced dimension, typically R^2 or R^3.')...
                    ];

                hFigure = figure('MenuBar','none','Name','Eestimate current source','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
                    'Resize','off','userData',0);
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
                if ~get(hFigure,'userData'), close(hFigure);return;end
                close(hFigure);
                drawnow
                val = g.GetPropertyValues();
                latency = val.latency;
                if length(latency) == 2, latency = latency(1):latency(2);end
                verbose = val.verbose;
                maxTol = val.maxTol;
                maxIter = val.maxIter;
                gridSize = val.gridSize;
                stateSpaceReducedDimension = val.stateSpaceReducedDimension;
                
                dispCommand = true;
            end
            if ~exist('maxTol','var'),   maxTol = 1e-3;end
            if ~exist('maxIter','var'),  maxIter = 100;end
            if ~exist('gridSize','var'), gridSize = 100;end
            if ~exist('verbose','var'),  verbose = true;end
            if ~exist('stateSpaceReducedDimension','var'), stateSpaceReducedDimension = 3;end
            stateSpaceReducedDimension(stateSpaceReducedDimension<2) = 2;
            stateSpaceReducedDimension(stateSpaceReducedDimension>3) = 3;
                        
            if dispCommand
                disp('Running:');
                fprintf('  mobilab.allStreams.item{%i}.stateSpaceGeometricAnalysis( latency, %f, %i, %i, %i, %i);\n',obj.container.findItem(obj.uuid),maxTol,maxIter,gridSize,verbose,stateSpaceReducedDimension);
            end
            
            % opening the surfaces by the Thalamus
            structName = {'Thalamus_L' 'Thalamus_R'};
            [~,K,L,rmIndices] = getSourceSpace4PEB(obj,structName);
            load(obj.surfaces);
            n = size(surfData(end).vertices,1); %#ok
            ind = setdiff(1:n,rmIndices);                     
            Nt = length(latency);

            options.maxTol = maxTol;
            options.maxIter = maxIter;
            options.gridSize = gridSize;
            options.verbose = verbose;
            
            commandHistory.commandName = 'stateSpaceGeometricAnalysis';
            commandHistory.uuid = obj.uuid;
            commandHistory.varargin{1} = latency;
            commandHistory.varargin{2} = maxTol;
            commandHistory.varargin{3} = maxIter;
            commandHistory.varargin{4} = gridSize;
            commandHistory.varargin{5} = verbose;
            commandHistory.varargin{6} = stateSpaceReducedDimension;            
            roiObj = createROIobj(obj,commandHistory);
            
            roiIndices = cell(roiObj.numberOfChannels,1);
            colorTable = unique(obj.atlas.colorTable);
            for it=1:length(colorTable), roiIndices{it} = obj.atlas.colorTable == colorTable(it);end
            
            %--
            [U,S,V] = svd(K/L,'econ');
            Ut = U';
            s2 = diag(S).^2;
            iLV = L\V;
            %--
            delta = 32;
            if size(roiObj,1) < 3*delta, delta=1;end
            J = zeros(size(surfData(end).vertices,1),2*delta); %#ok
            Y = obj.mmfObj.Data.x;
            X = roiObj.mmfObj.Data.x;
            obj.initStatusbar(1,Nt,'Estimating Primary Current Density...');
            [J(ind,1:delta),alpha,beta] = variationalDynLoreta(Ut,Y(latency(1:delta),:)',s2,iLV,L,[],[],options);
            for it=delta+1:delta:Nt
                if it+delta <= Nt
                    [J(ind,:),alpha,beta] = variationalDynLoreta(Ut,Y(latency(it-delta:it+delta-1),:)',s2,iLV,L,alpha,beta,options);
                    J(isnan(J)) = 0;
                    for jt=1:roiObj.numberOfChannels, X(it-delta:it+delta-1,jt) = median(J(roiIndices{jt},:))';end
                else
                    J = zeros(size(surfData(end).vertices,1),length(latency(it-delta:end))); %#ok
                    [J(ind,:),alpha,beta] = variationalDynLoreta(Ut,Y(latency(it-delta:end),:)',s2,iLV,L,alpha,beta,options);
                    J(isnan(J)) = 0;
                    for jt=1:roiObj.numberOfChannels, X(it-delta:end,jt) = median(J(roiIndices{jt},:))';end
                end
                obj.statusbar(it);
            end
            roiObj.mmfObj.Data.x = X;
            obj.statusbar(Nt);
            roiObj.stateSpaceGeometricAnalysis(stateSpaceReducedDimension,delta);
        end
        %%
        function estimateCurrentSourceDensityEPB(obj,latency)%#ok
            disp('Empirical Parametric Bayes.')
            disp('Not ready yet.')
        end
        %%
        function epochObj = epoching(obj,eventLabelOrLatency, timeLimits, channels, condition,subjectID,preStimulusLatency)
            % Creates epoch objects. Epoch objects don't result in a new addition to the tree.
            % Even when the epoch object manages its data through a memory mapped file they are
            % regarded as temporal variables, once the object is destroyed the associated binary
            % file is deleted.
            %
            % Input arguments:
            %       eventLabelOrLatency: if is a string or a cell array of strings it is used as
            %                            the event around to make the trials; if is a vector is
            %                            interpreted as the sample number of the events  around
            %                            to make the trials.
            %       timeLimits:          two elements vector specifying the size of the trial 
            %                            taking as reference the target latency. Ex: [-1 2]
            %                            correspond to one second before and two seconds after
            %                            the target event/latency.
            %       channels:            indices of the channels to epoch
            %       condition:           optional string specifying the name of the condition,
            %                            default: unknown
            %       subjectID:           unique identidier of the epoched data, obj.uuid or a 
            %                            combination of the former and obj.sessionUUID is recommended,
            %                            default: obj.sessionUUID
            %       preStimulusLatency:  two elements vector specifying the segment in the epoch 
            %                            considered pre-stimulus (or base line), default: [timeLimits(1) 0]
            %
            %
            % Output argument:
            %       epochObject:         handle to the epoched object
            %
            % Usage:
            %       eegObj     = mobilab.allStreams.item{ eegItem };
            %       eventType  = {'eventLabel_x'}; % for more than one type use: {'eventLabel_x', 'eventLabel_y', ...} 
            %       timeLimits = [-2 2];           % two seconds before and after the event (it does not have to be a symmetric window)
            %       channels   = 85;               % channel to epoch (could be more than one). Observe that in case of ICA data,
            %                                      % channels correspond to independent components activation.
            %       condition  = 'target1';        
            %       preStimulusLatency = [-1.7 -1.2]; % base-line: between 1.7 and 1.2 seconds before the stimulus/response
            %
            %       epObj = eegObj.epoching( eventType, timeLimits, channels, condition, eegObj.sessionUUID, preStimulusLatency);
            %       plot(epObj);
            %       [~,ersp,itc,frequency,time] = epObj.waveletTimeFrequencyAnalysis;
            %      
            % See eegEpoch for more details

            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 5, condition = 'unknown';end 
            if nargin < 6, subjectID = obj.uuid;end
            if nargin < 7, warning('MoBILAB:noChannels','Undefined preStimulusLatency, assuming half of the epoch.'); preStimulusLatency = [];end
            
            [data,time,eventInterval] = epoching@coreStreamObject(obj,eventLabelOrLatency, timeLimits, channels);
            
            if isempty(preStimulusLatency)
                preStimulusLatency = [timeLimits(1) 0];
            elseif any(preStimulusLatency < timeLimits(1)) || any(preStimulusLatency > timeLimits(end))
                preStimulusLatency = [timeLimits(1) 0];
            elseif length(preStimulusLatency) ~= 2
                error('preStimulusLatency must be a two elements vector.');
            end
            [~,loc1] = min(abs(time - preStimulusLatency(1)));
            [~,loc2] = min(abs(time - preStimulusLatency(2)));
            preStimulusLatency = [loc1 loc2];
            
            epochObj = eegEpoch(data,time,obj.label(channels),condition,eventInterval,subjectID,preStimulusLatency);
        end
        %%
        function epochObj = epochingTW(obj,latency, channels, condition,subjectID)
            % Creates epoch objects. Epoch objects don't result in new additions to the tree. 
            % Even when an epoch object manages its data through a memory mapped file they are
            % regarded as temporal variables, once the object is destroyed its binary file is
            % deleted. TW stands for time warping. This method makes trials between two set of
            % events/latencies, because each trial can have a slightly different size the resulting
            % trials are interpolated to a common time axis.
            % 
            % Input argument: 
            %       latency:     two columns matrix with the set of start end latencies (in seconds)
            %                    for each trial.
            %       channels:    indices of the channels to epoch
            %       condition:   optional string specifying the name of the condition, default: unknown
            %       subjectID:   unique identidier of the epoched data, obj.uuid or a combination of the
            %                    former and obj.sessionUUID is recommended, default: obj.sessionUUID
            %
            % Output arguments:
            %       epochObject: handle to the epoched object
            %
            % Usage:
            %       eegObj    = mobilab.allStreams.item{ eegItem };
            %       latency_walk = eegObj.event.getLatencyForEventLabel('701'); % Go event marker
            %       latency_stop = eegObj.event.getLatencyForEventLabel('702'); % Stop event marker
            %       latency = [latency_walk(:) latency_stop(:)];
            %       channels  = 85;             % channel to epoch (could be more than one). Observe that in case of ICA data,
            %                                   % channels correspond to independent components activation.
            %       condition  = 'walking';    
            %
            %       epObj      = eegObj.epochingTW(latency, channels, condition, eegObj.sessionUUID);
            %       plot(epObj);
            %       [~,ersp,itc,frequency,time] = epObj.waveletTimeFrequencyAnalysis;
            %      
            % See eegEpoch for more details
            
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 4, condition = 'unknown';end
            if nargin < 5, subjectID = obj.uuid;end 
            
            [data,time,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
            epochObj = eegEpoch(data,time,obj.label(channels),condition,eventInterval,subjectID);
        end
        %%
        function cobj = applyICAweights(obj,eegfile)
            dispCommand = false;
            cobj = [];
            if nargin < 2, eegfile = -1;end
            if ~ischar(eegfile)
                [FileName,PathName] = uigetfile2({'*.set','EEGLAB file'},'Select the .set file');
                if any([isnumeric(FileName) isnumeric(PathName)]), return;end
                eegfile = fullfile(PathName,FileName);
                dispCommand = true;
            end
            [path,name] = fileparts(eegfile);
            EEG = pop_loadset([name '.set'],path);
            if isempty(EEG.icaweights), error('This file does not contain ICA fields. Run ICA first.');end
            EEG.data = [];
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = [];
            metadata.parentCommand.commandName = 'applyICAweights';
            metadata.parentCommand.uuid = obj.uuid;
            metadata.parentCommand.varargin{1} = eegfile;
            
            labels = lower({EEG.chanlocs.labels})';
            [~,loc1,loc2] = intersect(lower(metadata.label),labels(EEG.icachansind),'stable');
            metadata.name = ['ica_' metadata.name];
            metadata.numberOfChannels = length(loc1);
            metadata.label = cell(metadata.numberOfChannels,1);
            for it=1:metadata.numberOfChannels, metadata.label{it} = ['IC' num2str(it)];end
            metadata.artifactMask = metadata.artifactMask(:,loc1);
            if ismac, [~,hash] = system('uuidgen'); else hash = java.util.UUID.randomUUID;end
            metadata.uuid = char(hash);
            metadata.binFile = fullfile(obj.container.mobiDataDirectory,[metadata.name '_' char(metadata.uuid) '.bin']);
            
            if isempty(obj.channelSpace)
                if ~isempty(EEG.chanlocs)
                    elec = [cell2mat({EEG.chanlocs.X})' cell2mat({EEG.chanlocs.Y})' cell2mat({EEG.chanlocs.Z}')];
                    metadata.channelSpace = elec(EEG.icachansind(loc2),:);
                    labels = {EEG.chanlocs.labels};
                    metadata.label = labels(EEG.icachansind(loc2));
                end
            else
                metadata.channelSpace = metadata.channelSpace(loc1,:);
            end
            
            data = obj.mmfObj.Data.x;
            obj.initStatusbar(1,metadata.numberOfChannels,'Copying data before applying ICA weights...');
            allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
            obj.statusbar(metadata.numberOfChannels);
            
            metadata.icasphere  = EEG.icasphere(loc2,loc2);
            metadata.icaweights = EEG.icaweights(loc2,:);
            metadata.icawinv    = EEG.icawinv(loc2,:);
            % metadata.icawinv    = pinv(metadata.icaweights*metadata.icasphere); %EEG.icawinv(loc2,:);
            % scaling = repmat(sqrt(mean(metadata.icawinv.^2))', [1 size(metadata.icaweights,2)]);
            % metadata.icaweights = metadata.icaweights.* scaling;
            %$ metadata.icawinv    = pinv(metadata.icaweights*metadata.icasphere);
            metadata.header = fullfile(obj.container.mobiDataDirectory,[metadata.name '_' char(metadata.uuid) '.hdr']);
            newHeader = metadata2headerFile(metadata);
            
            if dispCommand
                disp('Running:');
                disp(['  icaObj = mobilab.allStreams.item{ ' num2str(obj.container.findItem(obj.uuid)) ' }.applyICAweights( ''' eegfile ''' );']);
            end
            
            try
                cobj = icaEEG(newHeader);
                obj.container.item{end+1} = cobj;
                clear EEG
                
                if isfield(metadata,'segmentObj')
                    ind = cobj.getTimeIndex([cobj.segmentObj.startLatency(2:end) cobj.segmentObj.endLatency(1:end-1)]);
                    cobj.event = cobj.event.addEvent(ind,'boundary');
                    ind = cobj.getTimeIndex(cobj.segmentObj.endLatency);
                    cobj.event = cobj.event.addEvent(ind,'boundary');
                end
                
                clear metadata
                W = (cobj.icaweights*cobj.icasphere)';
                buffer_size = 1024;
                dim = size(cobj);
                obj.initStatusbar(1,dim(1),'Applying ICA weights...');
                for it=1:buffer_size:dim(1)
                    if it+buffer_size-1 > dim(1)
                        cobj.mmfObj.Data.x(it:end,:) = data(it:end,loc1)*W;
                        break
                    else
                        cobj.mmfObj.Data.x(it:it+buffer_size-1,:) = data(it:it+buffer_size-1,loc1)*W;
                    end
                    obj.statusbar(it);
                end
                obj.statusbar(dim(1));
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(obj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
    end
    methods(Hidden = true)
        function loadElectrodeWizard(obj,file)
            warning('This method will be deprecated, instead use ''readMontage'' with the same input arguments.');
            readMontage(obj,file);
        end
        %%
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid  = obj.uuid;
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
                case 'reReference'
                    prename = 'ref_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    channels2BeReferenced = commandHistory.varargin{1};
                    ind = ismember(obj.label,channels2BeReferenced);
                    metadata.label = obj.label(ind);
                    metadata.numberOfChannels = length(metadata.label);
                    if ~isempty(obj.channelSpace)
                        metadata.channelSpace = metadata.channelSpace(ind,:);
                    end
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    if exist(obj.leadFieldFile,'file')
                        metadata.leadFieldFile = fullfile(path,['lf_ ' metadata.name '_' metadata.uuid '.mat']);
                        load(obj.leadFieldFile);
                        K = K(ind,:); %#ok
                        save(metadata.leadFieldFile,'K');
                    end
                    metadata.artifactMask = obj.artifactMask(:,ind);
                    
                case 'artifactsRejection'
                    prename = 'artRej_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    copyfile(obj.binFile,metadata.binFile,'f');
                    % allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                case 'removeSubspace'
                    prename = 'rmSubs_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    copyfile(obj.binFile,metadata.binFile,'f');
                    
                case 'estimatePCD'
                    prename = 'pcd_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    metadata.timeStamp = obj.timeStamp(commandHistory.varargin{1});
                    
                    load(obj.surfaces);
                    load(obj.leadFieldFile);
                    dim = size(K); %#ok
                    n = size(surfData(end).vertices,1); %#ok
                    if n == dim(2)/3
                         metadata.numberOfChannels = n*3;
                    else metadata.numberOfChannels = n;
                    end
                    metadata.label = cell(metadata.numberOfChannels,1);
                    for it=1:metadata.numberOfChannels, metadata.label{it} = num2str(it);end
                    metadata.class = 'pcdStream';
                    allocateFile(metadata.binFile,obj.precision,[metadata.numberOfChannels length(metadata.timeStamp)]);
                    
                otherwise
                    error('Cannot make a copy of this object. Please provide a valid ''command history'' instruction.');
            end
            newHeader = metadata2headerFile(metadata);
        end
        %%
        function cobj = createROIobj(obj,commandHistory)
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            uuid = generateUUID;
            metadata.uuid = uuid;
            path = fileparts(obj.binFile);
            prename = 'roi_';
            metadata.name = [prename metadata.name];
            metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
            metadata.timeStamp = obj.timeStamp(commandHistory.varargin{1});
            metadata = rmfield(metadata,{'surfaces','leadFieldFile','atlas'});
            metadata.label = obj.atlas.label;
            I = setdiff(1:max(obj.atlas.colorTable),unique(obj.atlas.colorTable));
            metadata.label(I) = [];
            metadata.numberOfChannels = length(metadata.label);
            artifactMask = obj.artifactMask(commandHistory.varargin{1},:);
            artifactMask = sum(artifactMask,2);
            artifactMask = artifactMask./(max(artifactMask)+eps);
            metadata.artifactMask = repmat(artifactMask,1,metadata.numberOfChannels);
            metadata.class = 'roiStream';
            allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
            newHeader = metadata2headerFile(metadata);
            cobj = obj.container.addItem(newHeader);
        end
        %%
        function disp(obj)
            string = sprintf('  channelSpace:         <%ix3 double>',size(obj.channelSpace,1));
            if ~isempty(obj.surfaces)
                string = sprintf('%s\n  surfaces:             %s',string, obj.surfaces);
                string = sprintf('%s\n  atlas.colorTable:     <%ix1 %s>',string, length(obj.atlas.colorTable), class(length(obj.atlas.colorTable)));
                string = sprintf('%s\n  atlas.label:          <%ix1 cell>',string, length(obj.atlas.label));
            else
                string = sprintf('%s\n  surfaces:            ''''',string);
                string = sprintf('%s\n  atlas.colorTable:    []',string);
                string = sprintf('%s\n  atlas.label:         {[]}',string);
            end
            if ~isempty(obj.leadFieldFile)
                string = sprintf('%s\n  leadFieldFile:        %s',string, obj.leadFieldFile);
            else
                string = sprintf('%s\n  leadFieldFile:       ''''',string);
            end
            disp@coreStreamObject(obj)
            disp(string);
        end
        %%
        function properyArray = getPropertyGridField(obj)
            dim  = size(obj.channelSpace,1);
            if isempty(obj.atlas), 
                colorTable = '[]';
                roi = {''};
            else
                colorTable = ['<' num2str(length(obj.atlas.colorTable)) 'x1 ' obj.precision '>'];
                roi = obj.atlas.label;
            end
            properyArray1 = getPropertyGridField@coreStreamObject(obj);
            properyArray2 = [...
                PropertyGridField('channelSpace',['<' num2str(dim(1)) 'x3 ' obj.precision '>'],'DisplayName','channelSpace','ReadOnly',false,'Description','')...
                PropertyGridField('surfaces',obj.surfaces,'DisplayName','surfaces','ReadOnly',false,'Description','')...
                PropertyGridField('colorTable',colorTable,'DisplayName','atlas.colorTable','ReadOnly',false,'Description','')...
                PropertyGridField('roi',roi,'DisplayName','atlas.label','ReadOnly',false,'Description','')...
                PropertyGridField('leadFieldFile',obj.leadFieldFile,'DisplayName','leadFieldFile','ReadOnly',false,'Description','')...
                ];
            properyArray = [properyArray1 properyArray2];
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Add sensor locations');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'readMontage',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Create head model');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'buildHeadModelFromTemplate',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Compute lead field matrix');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'computeLeadFieldBEM',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Re-reference');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'reReference',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Remove aux-channels subspace');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'removeSubspace',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Artifact rejection');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'artifactsRejection',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'filter',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Clean line');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'cleanLine',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Apply ICA unmixing matrix');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'applyICAweights',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Run ICA');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'ica',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Time frequency analysis (CWT)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'continuousWaveletTransform',-1});
            jmenu.add(menuItem);
            %---------
            %menuItem = javax.swing.JMenuItem('Time frequency analysis (STFT)');
            %set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'shortTimeFourierTransform',-1});
            %jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Estimate Primary current density (PCD)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'estimatePCD',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('State space analysis');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'stateSpaceGeometricAnalysis',-1});
            jmenu.add(menuItem);

            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot spectrum');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrum',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot on scalp');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'plotOnScalp',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Show head model');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'plotHeadModel',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Inspect');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'inspect',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Export to EEGLAB');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'EEGstructure',0});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Annotation');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Generate batch script');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@generateBatch_Callback,obj});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="maroon">Delete object</HTML>');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});
            jmenu.add(menuItem);
        end
    end
    methods(Static), function triggerSaveHeader(~,evnt), saveHeader(evnt.AffectedObject, 'f'); end;end
end

%--
function [data,artP] = softMasking(data,mask,numberOfSamples)
dL = diff(logical(mask));
artP = full(mask);
latencyL = find(dL==1);
latencyR = find(dL==-1);

N = 2*round(numberOfSamples/2);
win = hann(2*N);
dim = size(data,1);
if ~isempty(latencyL)
    for it=1:length(latencyL), if latencyL(it)-N > 1, artP(latencyL(it)-N+1:latencyL(it)) = artP(latencyL(it)-N+1:latencyL(it)) + win(1:N);end;end
end
if ~isempty(latencyR)
    for it=1:length(latencyR), if latencyR(it)+N-1 < dim, artP(latencyR(it):latencyR(it)+N-1) = artP(latencyR(it):latencyR(it)+N-1)  + win(N+1:end);end;end
end
artP(artP>1) = 1;
data = bsxfun(@times,data,(1-artP));
end

%--
function copyMMF(EEG,streamObj)
% bytes = dir(streamObj.binFile);
% bytes = bytes.bytes/1e9;
% if bytes < 0.5
fid = fopen(EEG.data,'w');
if fid < 0, return;end
if ~iscell(streamObj), streamObj = {streamObj};end
N = length(streamObj{1}.timeStamp);
N2 = length(streamObj);
% precision = streamObj{1}.precision;
precision = 'single';
for it=1:N2, if ~strcmp(streamObj{it}.isMemoryMappingActive,'active'), fclose(fid);return;end;end
bufferSize = 1000;
streamObj{1}.container.container.initStatusbar(1,N,'Creating EEG.data...');
for it=1:bufferSize:N
    try writeThis = [];
        for jt=1:N2, writeThis = [writeThis;streamObj{jt}.data(it:it+bufferSize-1,:)'];end%#ok        
    catch writeThis = [];%#ok
        for jt=1:N2, writeThis = [writeThis;streamObj{jt}.data(it:end,:)'];end%#ok
    end
    fwrite(fid,writeThis(:),precision);
    streamObj{1}.container.container.statusbar(it);
end
streamObj{1}.container.container.statusbar(inf);
fclose(fid);
end
