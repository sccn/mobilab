classdef timeFrequencyStream < coreStreamObject
    properties(GetAccess = public, SetAccess = protected)
        frequency
    end
    properties(Dependent)
        power = [];
        phase = [];
    end
    properties(Hidden=true)
        svdFilter = 1;
    end
    methods
        %%
        function obj = timeFrequencyStream(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@coreStreamObject(header);
        end
        %%
        function frequency = get.frequency(obj)
            if isempty(obj.frequency), obj.frequency = retrieveProperty(obj,'frequency');end
            frequency = obj.frequency;
        end
        %%
        function power = get.power(obj)
            if obj.isMemoryMappingActive, power = obj.mmfObj.Data.x.^2+obj.mmfObj.Data.y.^2;end
        end
        function phase = get.phase(obj)
            if obj.isMemoryMappingActive, phase = atan( obj.mmfObj.Data.y./obj.mmfObj.Data.x);end
        end
        %%
        function connect(obj)
            if isempty(obj.binFile), return;end
            try 
                obj.mmfObj = memmapfile(obj.binFile,'Format',{obj.precision [length(obj.timeStamp) length(obj.frequency) obj.numberOfChannels] 'x';...
                    obj.precision [length(obj.timeStamp) length(obj.frequency) obj.numberOfChannels] 'y'},'Writable',obj.writable);
            catch ME
                obj.mmfObj = [];
                ME.rethrow;
            end
        end
        %%
        function disp(obj)
            if ~isvalid(obj); disp('Invalid or deleted object.');return;end     
            dim = size(obj);
            string = sprintf('\nClass:  %s\nProperties:\n  name:                 %s\n  uuid:                 %s\n  samplingRate:         %i Hz\n  timeStamp:            <1x%i double>\n  numberOfChannels:     %i\n  power:                <%ix%ix%i %s>\n  phase:                <%ix%ix%i %s>\n  coefficients:         <%ix%ix%i complex>\n  event.latencyInFrame: <1x%i double>\n  event.label:          <%ix1 cell>',...
                class(obj),obj.name,char(obj.uuid),obj.samplingRate,length(obj.timeStamp),obj.numberOfChannels,dim(1),dim(2),dim(3),obj.precision,dim(1),dim(2),dim(3),obj.precision,dim(1),dim(2),dim(3),length(obj.event.latencyInFrame),length(obj.event.label));
            try %#ok
                string = sprintf('%s\n  history:              %s',string,obj.history(1,:));
                for it=2:size(obj.history,1), string = sprintf('%s\n                        %s',string,obj.history(it,:));end
            end
            disp(string);
        end
        %%
        function dim = size(obj,d)
            if obj.isMemoryMappingActive
                dim = obj.mmfObj.Format{1,2};
            else
                dim = [0 0];
            end
            if nargin > 1, try dim = dim(d); catch dim = [];end;end %#ok
        end
        %%
        function browserObj = plot(obj), browserObj = spectrogramBrowserHandle(obj);end
        function browserObj = spectrogramBrowser(obj,defaults)
            if nargin < 2, defaults.browser  = @spectrogramBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser = @spectrogramBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
        %%
        function epochObj = mkEpoch(obj,eventLabelOrLatency, timeLimits, channels, condition)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 5, condition = 'unknownCondition';end 
            if iscellstr(eventLabelOrLatency)
                latency = obj.event.getLatencyForEventLabel(eventLabelOrLatency);
            elseif isvector(eventLabelOrLatency)
                latency = eventLabelOrLatency;
            else
                error('First argument has to be a cell array with the event labels or a vector with of latencies (in samples).');
            end
            
            Nf = length(obj.frequency);
            Nt = length(latency);
            t1 = timeLimits(1):1/obj.samplingRate:0;
            t2 = 1/obj.samplingRate:1/obj.samplingRate:timeLimits(2);
            time = [t1 t2];
            d1 = length(t1)-1;
            d2 = length(t2);
            data = zeros(length(time),Nt,length(channels));
            tfData = zeros(length(time),Nf,Nt,length(channels));
            
            index = obj.container.findItem(obj.parentCommand.uuid);
            streamObj = obj.container.item{index};
            
            
            if isa(streamObj,'icaStream')
                channelLabel = cell(length(channels),1);
                tmp = find(ismember(streamObj.label,obj.label(channels)));
                for it=1:length(channels), channelLabel{it} = ['IC ' num2str(tmp(it))];end
            else
                channelLabel = obj.label(channels);
            end
            
            channels = sort(channels);
            [~,loc] = intersect(streamObj.label,obj.label(channels),'stable');
            
            sdata = streamObj.data(:,loc);
            rmThis = zeros(Nt,1);
            tfDdataReal = obj.mmfObj.Data.x;
            tfDdataImag = obj.mmfObj.Data.y;
            for k=1:Nt
                try
                    data(:,k,:) = sdata([latency(k)-d1:latency(k) latency(k)+1:latency(k)+d2],:);
                    tfData(:,:,k,:) = tfDdataReal([latency(k)-d1:latency(k) latency(k)+1:latency(k)+d2],:,channels) +...
                        1i*tfDdataImag([latency(k)-d1:latency(k) latency(k)+1:latency(k)+d2],:,channels);
                catch %#ok
                    rmThis(k) = k;
                end
            end
            rmThis(rmThis==0) = [];
            if ~isempty(rmThis)
                data(:,rmThis,:) = [];
                tfData(:,:,rmThis,:) = [];
            end
            base = mean(obj.power(:,:,channels));
            if exist(streamObj.surfaces,'file')
                plotGCV = false;
                J = streamObj.estimateScalpMapSource(loc,plotGCV,99);
                headModelInfo.J = J;
                headModelInfo.V = streamObj.icawinv(:,loc);
                headModelInfo.channelLabel = streamObj.label;
                headModelInfo.channelSpace = streamObj.channelSpace;
                headModelInfo.surfaces = streamObj.surfaces;
                headModelInfo.atlas = streamObj.atlas;
            end
            try svdFilter = obj.svdFilter(:,:,channels); catch svdFilter = 1;end %#ok
            epochObj = tfEpoch('data',data,'time',time,'channelLabel',channelLabel,'condition',condition,'eventInterval',diff(latency)/obj.samplingRate,...
                'frequency',obj.frequency,'tfData',tfData,'headModelInfo',headModelInfo,'base',base,'svdFilter',svdFilter); %#ok
        end
        %%
        function svdDenoising(obj,maxPowerLost)
            if nargin < 2, maxPowerLost = 0.02;end
            power2keep = 1-maxPowerLost;
            if power2keep > 1 || power2keep < 0, power2keep = 0.98;end
            
            dim = size(obj);
            Nt = dim(1);Nf = dim(2);Nc=dim(3);
            k = 1000;
            Pmin = Nf*log(Nf);
            Nsample = round(k*Pmin);
            I = round(linspace(1,Nt,Nsample));
            while isempty(I)
                k = 0.5*k;
                if k < 10*Pmin; error('Not enough data to do SVD denoising.');end
                Nsample = round(k*Pmin);
                I = round(linspace(1,Nt,Nsample));
            end
            
            obj.svdFilter = zeros(Nf,Nf,Nc);
            obj.container.container.initStatusbar(1,Nc,'SVD denoising...');
            for it=1:Nc
                [~,S,V] = svds(log(obj.mmfObj.Data.x(I,:,it).^2+obj.mmfObj.Data.y(I,:,it).^2), Nf);   % taking the power
                %%
                s = diag(S);
%                 is = s;
%                 tol = 0.01*(max(s)-min(s));
%                 is(s > tol) = 1./s(s > tol);
%                 iS = diag(is);
%                 
                lrS = zeros(Nf,1);
                is  = lrS;
                n = find(cumsum(s.^2) / sum(s.^2) > power2keep, 1, 'first');
                lrS(1:n) = s(1:n);
                lrS = diag(lrS);
                is(1:n) = 1./s(1:n);
                iS = diag(is);
                H =  V*iS * lrS * V';
                obj.svdFilter(:,:,it) = H;
                obj.container.container.statusbar(it);
            end
        end
        %%
        function  jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            
            menuItem = javax.swing.JMenuItem('Plot spectrogram');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrogramBrowser',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
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
        function newHeader = createHeader(obj,commandHistory) %#ok
            disp('Not implemented yet.')
        end
    end
end