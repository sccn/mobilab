% Definition of the class dataStream. This class serves as base class to 
% all objects holding any modality of time series data. It defines methods 
% for re-sampling, smoothing, filtering, plotting, fft and time frequency
% analysis.
%
% For more details visit: https://code.google.com/p/mobilab/ 
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Apr-2011

%%
classdef dataStream < coreStreamObject
    methods
        %%
        function obj = dataStream(header)
            % Creates a dataStream object.
            % 
            % Input arguments:
            %       header:     header file (string)
            % 
            % Output arguments:
            %       obj:        dataStream object (handle)
            %
            % Usage:
            %       obj = dataStream(header);
            
            if nargin < 1, error('Not enough input arguments.');end
            obj@coreStreamObject(header);
        end
        %%
        function epochObj = epoching(obj,eventLabelOrLatency, timeLimits, channels, condition)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noTImeLimits','Undefined time limits, assuming [-1 1] seconds.'); timeLimits = [-1 1];end
            if nargin < 4, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 5, condition = 'unknown';end 
            
            [data,time,eventInterval] = epoching@coreStreamObject(obj,eventLabelOrLatency, timeLimits, channels);
            epochObj = streamEpoch('data',data,'time',time,'channelLabel',obj.label(channels),'condition',condition,'eventInterval',eventInterval);
        end
        function epochObj = epochingTW(obj,latency, channels, condition)
            if nargin < 2, error('Not enough input arguments.');end
            if nargin < 3, warning('MoBILAB:noChannels','Undefined channels to epoch, epoching all.'); channels = 1:obj.numberOfChannels;end
            if nargin < 4, condition = 'unknownCondition';end
            
            switch class(obj)
                case 'eeg'
                    [data,time,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
                    epochObj = eegEpoch('data',data,'time',time,'channelLabel',obj.label(channels),'condition',condition,'eventInterval',eventInterval);
                case {'mocap' 'pcaMocap'}
                    if isa(obj,'mocap'), I = reshape(1:obj.numberOfChannels,[2 obj.numberOfChannels/2]);
                    else I = reshape(1:obj.numberOfChannels,[3 obj.numberOfChannels/3]);
                    end
                    channels = I(:,channels);
                    channels = channels(:);
                    [xy,time,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
                    [n,m,p] = size(xy);
                    xy = squeeze(mean(xy,2));
                    children = obj.children;
                    if ~isempty(children)
                        derivativeLabel = cell(length(children),1);
                        data = zeros(n,p*length(children),m);
                        indices = reshape(1:p*length(children),[p length(children)]);
                        for it=1:length(children)
                            data(:,indices(:,it),:) = permute( epochingTW@coreStreamObject(children{it},latency, channels), [1 3 2] );
                            if ~isempty(strfind(children{it}.name,'vel')),       derivativeLabel{it} = 'Velocity';
                            elseif ~isempty(strfind(children{it}.name,'acc')),   derivativeLabel{it} = 'Acceleration';
                            elseif ~isempty(strfind(children{it}.name,'jerk')),  derivativeLabel{it} = 'Jerk';
                            else                                                 derivativeLabel{it} = ['Dt' num2str(it)];
                            end
                        end
                    end
                    epochObj = mocapEpoch('data',data,'time',time,'channelLabel',obj.label(channels),'condition',condition,'eventInterval',eventInterval,'xy',xy,'derivativeLabel',derivativeLabel);
                otherwise
                    [data,time,eventInterval] = epochingTW@coreStreamObject(obj,latency, channels);
                    epochObj = streamEpoch('data',data,'time',time,'channelLabel',obj.label(channels),'condition',condition,'eventInterval',eventInterval);
            end
        end
        %%
        function cobj = sgolayFilter(obj,varargin)
            % Implements and applies a Savitzky-Golay filter. Savitzky-Golay smoothing filters
            % (also called digital smoothing polynomial filters or least-squares smoothing filters)
            % are typically used to "smooth out" a noisy signal whose frequency span (without noise)
            % is large. In this type of application, Savitzky-Golay smoothing filters perform much
            % better than standard averaging FIR filters, which tend to filter out a significant
            % portion of the signal's high frequency content along with the noise. Although
            % Savitzky-Golay filters are more effective at preserving the pertinent high frequency
            % components of the signal, they are less successful than standard averaging FIR filters
            % at rejecting noise. Savitzky-Golay filters are optimal in the sense that they minimize
            % the least-squares error in fitting a polynomial to frames of noisy data.
            %
            % Input arguments:
            %       order:       order of the polynomial
            %       movingWindow: number of samples used in the least-square fitting of the polinomial.
            %                    The size of the window must be odd, if order = movinWindow - 1. the 
            %                    filter produce no smoothing.
            %         
            % Output arguments:  
            %       cobj:       handle to the new object
            %
            % Usage:
            %        order = 4;
            %        movingWindow = 33;
            %        obj  = mobilab.allStreams.item{ itemIndex };
            %        cobj = obj.sgolayFilter( order, movingWindow );
            %        figure;plot( obj.timeStamp, [obj.data(:,1) cobj.data(:,1)] );
            %        xlabel('Time (sec)');
            %        legend({obj.name cobj.name});
            
            dispCommand = false;
            if  isnumeric(varargin{1}) && length(varargin{1}) == 1 && varargin{1} == -1
                desc = 'Savitzky-Golay smoothing filters (also called digital smoothing polynomial filters or least-squares smoothing filters) are typically used to "smooth out" a noisy signal whose frequency span (without noise) is large. In this type of application, Savitzky-Golay smoothing filters perform much better than standard averaging FIR filters, which tend to filter out a significant portion of the signal''s high frequency content along with the noise. Although Savitzky-Golay filters are more effective at preserving the pertinent high frequency components of the signal, they are less successful than standard averaging FIR filters at rejecting noise. Savitzky-Golay filters are optimal in the sense that they minimize the least-squares error in fitting a polynomial to frames of noisy data.';             
                prefObj = [...
                    PropertyGridField('order',4,'DisplayName','Order','Description',sprintf('%s\nThe polynomial order k must be less than the frame size.',desc))...
                    PropertyGridField('frameSize',33,'DisplayName','Movin window','Description','Number of samples used in the least-square fitting of the polinomial. The size of the window must be odd, if Order = movinWindow - 1. the filter produce no smoothing.')...
                    ];

                hFigure = figure('MenuBar','none','Name','Data denoising','NumberTitle', 'off','Toolbar', 'none','Units','pixels','Color',obj.container.container.preferences.gui.backgroundColor,...
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
                varargin{1} = val.order;
                varargin{2} = val.frameSize;
                dispCommand = true;
            end
            
            if length(varargin) < 1, order = 4; else order = varargin{1};end
            if length(varargin) < 2, frameSize = 33; else frameSize = varargin{2};end
            
            try
                commandHistory.commandName = 'sgolayFilter';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = order;
                commandHistory.varargin{2} = frameSize;
                
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                data = obj.mmfObj.Data.x;
                obj.initStatusbar(1,cobj.numberOfChannels,'Filtering...');
                for it=1:cobj.numberOfChannels
                    cobj.mmfObj.Data.x(:,it)  = sgolayfilt(data(:,it),order,frameSize);
                    obj.statusbar(it);
                end
                % cobj.mmfObj.Data.x = sgolayfilt(data,order,frameSize);
                
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(cobj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function cobj = smooth(obj,varargin)
            % Smooth the data.
            % 
            % Input arguments:
            %       movingWindow: length of the smoothing window (integer), 
            %                     default: 16
            %       method:       method of smoothing, can be moving, lowess,
            %                     loess, sgolay, rlowess, rloess; default: moving
            %       channels:     channels to smooth, default: all
            % 
            % Output arguments:
            %       cobj: handle to the new object
            %
            % Usage:
            %        movingWindow = 32;
            %        method = 'moving';
            %        obj  = mobilab.allStreams.item{ itemIndex };
            %        cobj = obj.smooth( movingWindow, method );
            %        figure;plot( obj.timeStamp, [obj.data(:,1) cobj.data(:,1)] );
            %        xlabel('Time (sec)');
            %        legend({obj.name cobj.name});
            
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if varargin{1} == -1
                prompt = {'Enter the length of the window',...
                    'Enter the method (''moving'',''lowess'',''loess'',''sgolay''):'};
                dlg_title = 'Smooth';
                num_lines = 1;
                def = {'32','moving'};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = str2double(varargin{1});
                if isnan(varargin{1}), varargin{1} = '';end
                dispCommand = true;
            end
            if length(varargin) < 1
                 movingWindow = 16;
            else movingWindow = varargin{1};
            end
            if length(varargin) < 2
                 method = 'moving';
            else method = varargin{2};
            end    
            if length(varargin) < 3
                 channels = 1:obj.numberOfChannels;
            else channels = varargin{3};
            end
            
            if ~isnumeric(movingWindow), error('prog:input','First argument must be the length of the moving window.');end
            if ~ischar(method), error('prog:input','Second argument must be a string that specify the soothing method.');end
            switch lower(method)
                case 'moving'
                case 'lowess'
                case 'loess'
                case 'sgolay'
                case 'rlowess'
                case 'rloess'
                otherwise, error('prog:input','Unknown smoothing method. Go to smooth help page to see the alternatives.');
            end
            if ~isnumeric(channels), error('Invalid input argument.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid input argument.');end
            Nch = length(channels);
            fprintf('allDataStreams.item{%i}.smooth(%i,''%s'');\n',obj.container.findItem(obj.uuid),movingWindow,method);
            try 
                commandHistory.commandName = 'smooth';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = movingWindow;
                commandHistory.varargin{2} = method;
                commandHistory.varargin{3} = channels;
                
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    disp(['  ' cobj.history]);
                end
                
                cobj.mmfObj.Writable = true;
                obj.initStatusbar(1,Nch,'Smoothing...');
                for it=1:Nch
                    cobj.mmfObj.Data.x(:,it)  = smooth(obj.timeStamp(:),obj.mmfObj.Data.x(:,channels(it)),...
                        movingWindow,method);
                    obj.statusbar(it);
                end
                cobj.mmfObj.Writable = false; 
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(cobj.container.findItem(cobj.uuid));end
                ME.rethrow;
            end
        end
        %%
        function [b, hFigure] = firDesign(obj,order,varargin)
            % Designs a windowed linear-phase FIR filter using a Hann (Hanning)
            % window. It uses internally the function fir1.
            % 
            % Input arguments:
            %       filterOrder: integer representing the order of the filter
            %       filterType:  could be: 1) 'lowpass', 2) highpass, 3) 'bandpass',
            %                    or 4) 'stopband'
            %       cutoff:      vector of cutoff frequencies (in Hz)
            %       plotFlag:    logical that if true plots the frequency response of the filter
            %       
            % Output arguments:
            %       b:           coefficients of the filter 
            % 
            % Usage:
            %        eegObj = mobilab.allStreams.item{ eegItem };
            %        filterOrder = round(eegObj.samplingRate*1.25);
            %        cutoff      = [1 eegObj.samplingRate/4];
            %        b = eegObj.firDesign( filterOrder, 'bandpass', cutoff, true);
            
            if nargin < 2, error('Not enough input arguments.');end
            N    = round(order/2)*2; % Filter order
            flag = 'scale';          % Sampling Flag
            win  = hann(N+1);        % Create a Hann window
            filterType = lower(varargin{1});
            if length(varargin) > 2, plotFreqz = varargin{3}; else plotFreqz = false;end
            switch filterType
                case 'lowpass'
                    if length(varargin{2}) ~= 1, error('The second argument must be a number with the cutoff frequency.');end
                    Fpass = varargin{2};     % Passband Frequency
                    b     = fir1(N, Fpass/(obj.samplingRate/2), 'low', win, flag); % Calculate the coefficients using the FIR1 function.
                    
                case 'highpass'
                    if length(varargin{2}) ~= 1, error('The second argument must be a number with the cutoff frequency.');end
                    Fpass = varargin{2};    % Passband Frequency
                    b     = fir1(N, Fpass/(obj.samplingRate/2), 'high', win, flag);
                    
                case 'bandpass'
                    if length(varargin{2}) ~= 2, error('The second argument must be a vector with the cutoff frequencies (e.g., [Fc1 Fc2])');end
                    Fpass1 = varargin{2}(1);  % First Passband Frequency
                    Fpass2 = varargin{2}(2);  % Second Passband Frequency
                    b      = fir1(N, [Fpass1 Fpass2]/(obj.samplingRate/2), 'bandpass', win, flag);
                    
                case 'bandstop'
                    if length(varargin{2}) ~= 2, error('Bandstop filter needs two cutoff frequencies (e.g., [Fstop1 Fstop2])');end
                    Fstop1 = varargin{2}(1);  % First Stopband Frequency
                    Fstop2 = varargin{2}(2);  % Second Stopband Frequency
                    b      = fir1(N, [Fstop1 Fstop2]/(obj.samplingRate/2), 'stop', win, flag);
                    
                otherwise, error('Invalid type of filter. Try one of these: ''lowpass'', ''highpass'', ''bandpass'', or ''bandstop''.');
            end
            hFigure = [];
            if plotFreqz
                hFigure = figure('MenuBar','none','Toolbar','figure','Color',obj.container.container.preferences.gui.backgroundColor,'userData',1);
                hb = uicontrol(hFigure,'Position',[365 6 70 21],'String','Cancel','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@cancelCallback);
                set(hb,'Units','Normalized')
                hb = uicontrol(hFigure,'Position',[443 6 70 21],'String','Ok','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                    'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@okCallback);
                set(hb,'Units','Normalized')
                freqz(b,1,[],obj.samplingRate);
                uiwait(hFigure);
                if ~ishandle(hFigure),       hFigure = [];end
                val = ~get(hFigure,'userData');
                close(hFigure);
                if val, hFigure = [];end
            end
        end
        %%
        function cobj = filter(obj,varargin)
            % Designs and applies or applies a zero-lag FIR filter.
            % 
            % Input arguments: (Variant 1)
            %       filterType:   can be: 1) lowpass, 2) bandpass, 3) highpass,
            %                     or 4) bandstop
            %       cutOff:       vector of cutoff frequencies (in Hz)
            %       channels:     channels to filter, default: all
            %       order:        order of the filter, default: (1+1/4)*samplingRate
            %       plotFreqz:    if true plots the frequency response of the filter
            % 
            % Input arguments: (Variant 2)
            %       b: filter's numerator's coefficients
            %       a: filter's denominator's coefficients 
            %
            % Output arguments:
            %       cobj:   handle to the new object
            %
            % Usage:
            %        eegObj = mobilab.allStreams.item{ eegItem };
            %
            %        % Variant 1:
            %        eegObjFilt1 = eegObj.filter( 'bandpass', [1 45] );
            %
            %        % Variant 2:
            %        % only computes the coefficient of the filter
            %        b = eegObj.firDesign( 640, 'bandpass', [1 45] );
            %        eegObjFilt2  = eegObj.filter( b, 1 );
            %        figure;plot( eegObj.timeStamp, [eegObj.data(:,1) eegObjFilt1.data(:,1)] );
            %        xlabel('Time (sec)'); legend({eegObj.name eegObjFilt1.name});
            %        figure;plot( eegObj.timeStamp, eegObjFilt1.data(:,1)-eegObjFilt2.data(:,1) );
            %        xlabel('Time (sec)');ylabel('Variant1 - Variant2')
            
            cobj = [];
            dispCommand = false;
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if nargin < 2, error('prog:input','Not enough input arguments.');end
            if isnumeric(varargin{1}) && length(varargin{1}) ==1 && varargin{1} == -1
                properyArray = {...
                    {'Type (lowpass, bandpass, highpass, or bandstop)','CutOff','Channels','Order','Freqz'},...
                    {'bandpass',['[1 ' num2str(obj.samplingRate/4) ']'], ['[' num2str(1:obj.numberOfChannels) ']'],num2str(obj.samplingRate*4),'false'}};
                params = inputdlg(properyArray{1},'Filter',1,properyArray{2});
                if isempty(params)
                    return;
                end
                varargin{1} = params{1};
                varargin{2} = str2num(params{2});
                varargin{3} = str2num(params{3});
                varargin{4} = str2num(params{4});
                varargin{5} = str2num(params{5});
                dispCommand = true;
            end
            plotFreqz = false;
            if length(varargin) == 5, if varargin{5}, plotFreqz = varargin{5};end;end
            if ischar(varargin{1})
                if length(varargin) < 4
                    N = obj.samplingRate*1.25;
                elseif isnumeric(varargin{4}) && length(varargin{4}) == 1
                     N = varargin{4};
                else N = obj.samplingRate*1.25;
                end
                a = 1;
                [b, hFigure] = firDesign(obj,N,varargin{1},varargin{2},plotFreqz);
                if plotFreqz
                    if isempty(hFigure), return;end
                end
                outMsg = sprintf('  mobilab.allStreams.item{%i}.filter( ''%s'', [%s]);\n',obj.container.findItem(obj.uuid),varargin{1},num2str(varargin{2}));
            else
                if isnumeric(varargin(1))
                    b = varargin(1);
                elseif iscell(varargin(1))
                    if isnumeric(varargin{1}), b = varargin{1};end
                else error('First argument must be the numerator coefficients of the filter.');
                end
                if isnumeric(varargin(2)), a = varargin(2);
                elseif iscell(varargin(2)), 
                    if isnumeric(varargin{2}), a = varargin{2};end
                else error('Second argument must be the denominator coefficients of the filter.');
                end
                if plotFreqz
                    hFigure = figure('MenuBar','none','Toolbar','figure','Color',obj.container.container.preferences.gui.backgroundColor,'userData',1);
                    hb = uicontrol(hFigure,'Position',[365 6 70 21],'String','Cancel','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                        'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@cancelCallback);
                    set(hb,'Units','Normalized')
                    hb = uicontrol(hFigure,'Position',[443 6 70 21],'String','Ok','ForegroundColor',obj.container.container.preferences.gui.fontColor,...
                        'BackgroundColor',obj.container.container.preferences.gui.buttonColor,'Callback',@okCallback);
                    set(hb,'Units','Normalized')
                    freqz(b,a,[],obj.samplingRate);
                    uiwait(hFigure);
                    if ~ishandle(hFigure), return;end
                    userData = get(hFigure,'userData');
                    close(hFigure);
                    if ~userData, disp('Aborting');return;end    
                end
                outMsg = sprintf('\n  allDataStreams.item{%i}.filter( b, a);\n',obj.container.findItem(obj.uuid));
            end
            if length(varargin) >= 3
                 channels = varargin{3};
            else channels = 1:obj.numberOfChannels;
            end
            if ~isnumeric(channels), error('Invalid input argument.');end
            if ~all(intersect(channels,1:obj.numberOfChannels)), error('Invalid input argument.');end
            
            Nch = length(channels);
            try
                commandHistory.commandName = 'filter';
                commandHistory.uuid        = obj.uuid;
                commandHistory.varargin{1} = b;
                commandHistory.varargin{2} = a;
                commandHistory.varargin{3} = channels;
                
                cobj = obj.copyobj(commandHistory);
                
                if dispCommand
                    disp('Running:');
                    %disp(['  ' cobj.history]);
                    fprintf('%s',outMsg);
                end
                cobj.mmfObj.Writable = true;
                try
                    data = obj.mmfObj.Data.x;
                    obj.initStatusbar(1,Nch,'Filtering...');
                    delta = 4*maxNumCompThreads;
                    for it=1:delta:Nch
                        if it+delta<=Nch
                            cobj.mmfObj.Data.x(:,it:it+delta) = filtfilt_fast(b,a,data(:,channels(it:it+delta)));
                        else
                            cobj.mmfObj.Data.x(:,it:end) = filtfilt_fast(b,a,data(:,channels(it:end)));
                        end
                        obj.statusbar(it);
                    end
                    obj.statusbar(Nch);
                catch ME
                    obj.statusbar(Nch);
                    ME.rethrow;
                end
                cobj.mmfObj.Writable = false;
            catch ME
                if exist('cobj','var'), obj.container.deleteItem(cobj.container.findItem(cobj.uuid));end
                if strcmp(ME.message,'Data must have length more than 3 times filter order.')
                    error('prog:input','The data must have length more than 3 times filter order.\nTry fdatool to redefine the filter and call this method passing the new coefficients.');
                end
                ME.rethrow;
            end
        end
        %%
        function cobj = resample(obj,newSamplingRate,method,flag)
            % Re-sample the time series. This method is implemented in two steps:
            % 1) interpolation to match the new sampling rate and 2) lowpass filtering
            % to eliminate possible aliasing. You can change the sampling rate of the
            % time series simply by changing obj.samplingRate, the set method of this
            % property will call resample to do the job.
            % 
            % Input arguments:
            %       newSamplingRate: new sampling rate
            %       
            % Output arguments:
            %       cobj:            handle to the new object
            %         
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       
            %       % downsampling
            %       eegObj1 = eegObj.resample( eegObj.samplingRate/2 );
            %
            %       % upsampling
            %       eegObj2 = eegObj.resample( eegObj.samplingRate*2 );
            %
            %       figure; hold on; grid on;
            %       plot( eegObj.timeStamp, eegObj.data(:,1));
            %       plot(eegObj1.timeStamp, eegObj1.data(:,1), 'g');
            %       plot(eegObj2.timeStamp, eegObj2.data(:,1), 'r');
            %       xlabel('Time (sec)'); legend({eegObj.name eegObj1.name eegObj2.name});
            
            if nargin < 2, error('You must enter the new sampling rate.');end
            if newSamplingRate == -1
                params = inputdlg({'New sampling rate'},'Resample',1,{num2str(obj.samplingRate)});
                if isempty(params), return;end
                newSamplingRate = str2num(params{1});
            end
            if nargin < 3, method = obj.container.container.preferences.eeg.resampleMethod;end
            if isempty(method), method = obj.container.container.preferences.eeg.resampleMethod;end
            if nargin < 4, flag = 0;end
            if flag && ~obj.writable, error('MoBILAB:attempt_to_delete_read_only_object','Cannot modify raw data. try cobj = copyobj(obj); before re-sampling this stream.');end
            if ~isnumeric(newSamplingRate),
                error('prog:input','First argument must be the new sample rate.');
            end
            if ~ischar(method),
                error('prog:input','Third argument must be a string that specify the interpolation method.');
            end
            switch lower(method)
                case 'nearest'
                case 'linear'
                case 'spline'
                case 'pchip'
                case 'v5cubic'
                otherwise
                    error('prog:input','Unknown interpolation method. Go to interp1 help page to see the alternatives.');
            end
            
            timeStampi = interp1(1:length(obj.timeStamp), obj.timeStamp',1:obj.samplingRate/newSamplingRate:length(obj.timeStamp),method,'extrap');
            if obj.isMemoryMappingActive
                try
                    commandHistory.commandName = 'resample';
                    commandHistory.uuid        = obj.uuid;
                    commandHistory.varargin{1} = newSamplingRate;
                    commandHistory.varargin{2} = method;
                    
                    cobj = obj.copyobj(commandHistory);
                    cobj.mmfObj.Writable = true;
                    data = obj.mmfObj.Data.x;
                    if newSamplingRate < obj.samplingRate
                        b = fir1(min([512 round(newSamplingRate*1.25)]), 0.99, 'low');
                        for it=1:obj.numberOfChannels
                            tmp = interp1(obj.timeStamp',data(:,it),timeStampi(:),method,'extrap');
                            cobj.mmfObj.Data.x(:,it) = filtfilt_fast(b,1,tmp);
                        end
                    else
                        cobj.mmfObj.Data.x = interp1(obj.timeStamp',data,timeStampi(:),method,'extrap');
                    end
                    cobj.mmfObj.Writable = false;
                    if flag
                        obj.disconnect;
                        tmp_binFile = obj.binFile;
                        obj.binFile = cobj.binFile;
                        obj.samplingRate = cobj.samplingRate;
                        obj.timeStamp = timeStampi;
                        saveHeader(obj,'f');
                        obj.connect;
                        obj.event = cobj.event;
                        cobj.binFile = tmp_binFile;
                        saveHeader(obj,'f');
                        obj.container.deleteItem(cobj.uuid);
                    end
                catch ME
                    if exist('cobj','var')
                        obj.container.deleteItem(obj.container.findItem(cobj.uuid));
                    end
                    ME.rethrow;
                end
            end
        end
        
        %%
        function [psdData,frequency] = spectrum(obj,varargin)
            % Computes the power spectral density (psd).
            %
            % Input arguments:
            %       method:   can be mtm (Thomson multitaper method), welch,
            %                 periodogram, or yulear (Yule-Walker's method)
            %       channels: channels to compute the psd on, default: all
            %       plotFlag: if true plots the psd, default: true
            %       plotType: 2 or 3D plot, default: 2D
            %
            % Output arguments:
            %       psdData:   power spectral density (matrix)
            %       frequency: frequency axis (hertz)
            %
            % Usage:
            %       eegObj = mobilab.allStreams.item{ eegItem };
            %       method   = 'mtm'; 
            %       channels = 1:10;
            %       plotFlag = true;
            %       [psdData,frequency] = spectrum( eegObj, method, channels, plotFlag);
            
            dispCommand = false;
            if ~isempty(varargin) && varargin{1}(1) == -1
                properyArray = {...
                    {'Method (mtm , welch, periodogram, or yulear)','Channels','Plot','Plot in 3D'},...
                    {'welch',['[' num2str(1:obj.numberOfChannels) ']'],'true','false'}};
                params = inputdlg(properyArray{1},'Filter',1,properyArray{2});
                if isempty(params)
                    return;
                end
                varargin{1} = params{1};
                varargin{2} = str2num(params{2});
                varargin{3} = str2num(params{3});
                if str2num(params{4}), varargin{4} = '3D';else varargin{4} = '2D';end
                dispCommand = true;
            end
            if dispCommand
                disp('Running:');
                fprintf('  mobilab.allStreams.item{%i}.spectrum( ''%s'' ,[ %s ], plotFlag, plotType )\n',obj.container.findItem(obj.uuid),...
                    num2str(varargin{1}),num2str(varargin{2}));
            end
            
            if length(varargin) < 1, method = 'welch';else method = varargin{1};end
            if length(varargin) < 2, channels = 1:obj.numberOfChannels;else channels = varargin{2};end
            if length(varargin) < 3, plotFlag = false;else plotFlag = varargin{3};end
            if length(varargin) < 4, plotType = '2D'; else plotType = varargin{4};end
                
            channels(channels<1 | channels > obj.numberOfChannels) = [];
            Hs = eval(['spectrum.' method ';']);
            switch method
                case 'welch',       Hs.SegmentLength = obj.samplingRate*2;
                case 'mtm',         Hs.TimeBW = 3.5;
                case 'yulear',      Hs.Order = 128;
                case 'periodogram', Hs.WindowName = 'Hamming';
            end
            Nch = length(channels);
            fmin = 1;
            fmax = 0.8*obj.samplingRate/2-mod(0.8*obj.samplingRate/2,10);
            obj.initStatusbar(1,Nch,'Computing PSD...');
            data = obj.mmfObj.Data.x;
            psdObj = Hs.psd(data(:,channels(1)),'Fs',obj.samplingRate,'NFFT',2048);
            [~,loc1] = min(abs(psdObj.frequencies-fmin));
            [~,loc2] = min(abs(psdObj.frequencies-fmax));
            frequency = psdObj.frequencies(loc1:loc2);
            psdData = psdObj.Data(loc1:loc2);
            psdData = [psdData zeros(length(psdData),Nch-1)];
            
            %--
            y = sign(diff(psdData));
            if abs(sum(y(2:end))) <= 2
                warning('MoBILAB:badSpectrum','Weak estimation of the Power Spectral Density. Artifact removal is recomended before estimating the PSD.');
            end
            %--
            
            obj.statusbar(1);
            for it=2:Nch
                psdObj = Hs.psd(data(:,channels(it)),'Fs',obj.samplingRate,'NFFT',2048);
                psdData(:,it) = psdObj.Data(loc1:loc2);
                obj.statusbar(it);
            end
            if plotFlag
                figure('Toolbar','figure','Color',obj.container.container.preferences.gui.backgroundColor);
                if ~ischar(plotType), plotType = '2D';end
                
                if strcmp(plotType,'3D')
                    hold on
                    color = lines(length(channels));
                    One = ones(length(frequency),1);
                    for it=1:length(channels), plot3(frequency,it*One,10*log10(psdData(:,it)),'Color',color(it,:),'ButtonDownFcn',...
                            ['disp(''' obj.label{channels(it)} ''')'],'LineSmoothing','on');end    
                    zlabel('Power/frequency (dB/Hz)')
                    ylabel('Channels')
                    set(gca,'YTickLabel',obj.label(channels),'YTick',1:length(channels))
                    if length(channels) > 1, view(18,24); else view(0,0);end    
                else
                    h = plot(frequency,10*log10(psdData),'ButtonDownFcn','get(gco,''userData'')');
                    tmpLabels = obj.label(channels(:));
                    set(h(:),{'userData'},flipud(tmpLabels(:)));
                    ylabel('Power/frequency (dB/Hz)')
                end
                if ~isMatlab2014b(), set(h,'LineSmoothing','on');end
                xlabel('Frequency (Hz)')
                title([Hs.EstimationMethod ' Power Spectral Density Estimate']);
                grid on;
            end
        end
        %%
        function browserObj = plot(obj)
            % Plots the time series in a browser window.
            browserObj = dataStreamBrowser(obj);
        end
    end
    methods(Hidden = true)
        %% overloaded operators
        function cobj = minus(obj,obj2)
            commandHistory.commandName = 'minus';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin    = {obj2.uuid};
            if ~isa(obj2,'coreStreamObject'), error('Second argument must be an object defined within MoBILAB''s toolbox.');end
            if any(size(obj) ~= size(obj2)), error('Arguments must have the same dimensions.');end
            cobj = obj.copyobj(commandHistory);
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x - obj2.mmfObj.Data.x;
        end
        function cobj = plus(obj,obj2)
            commandHistory.commandName = 'plus';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin    = {obj2.uuid};
            if ~isa(obj2,'coreStreamObject'), error('Second argument must be an object defined within MoBILAB''s toolbox.');end
            if any(size(obj) ~= size(obj2)), error('Arguments must have the same dimensions.');end
            cobj = obj.copyobj(commandHistory);
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x + obj2.mmfObj.Data.x;
        end
        function cobj = times(obj,obj2)
            commandHistory.commandName = 'plus';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin    = {obj2.uuid};
            if ~isa(obj2,'coreStreamObject'), error('Second argument must be an object defined within MoBILAB''s toolbox.');end
            if any(size(obj) ~= size(obj2)), error('Arguments must have the same dimensions.');end
            cobj = obj.copyobj(commandHistory);
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x .* obj2.mmfObj.Data.x;
        end
        function cobj = rdivide(obj,obj2)
            commandHistory.commandName = 'plus';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin    = {obj2.uuid};
            if ~isa(obj2,'coreStreamObject'), error('Second argument must be an object defined within MoBILAB''s toolbox.');end
            if any(size(obj) ~= size(obj2)), error('Arguments must have the same dimensions.');end
            commandHistory.varargin{1} = obj.container.findItem(obj2.uuid);
            cobj = obj.copyobj(commandHistory);
            cobj.mmfObj.Data.x = obj.mmfObj.Data.x ./ (obj2.mmfObj.Data.x+eps);
        end
        %%
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid = obj.uuid;
            end
            
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            uuid = generateUUID;
            metadata.uuid = uuid;
            path = fileparts(obj.binFile);
            
            switch commandHistory.commandName
                case 'resample'
                    prename = 'resample_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    metadata.samplingRate = commandHistory.varargin{1};
                    metadata.timeStamp = interp1(1:length(obj.timeStamp), obj.timeStamp',1:obj.samplingRate/metadata.samplingRate:length(obj.timeStamp),'linear','extrap');
                    eventObj = obj.event.interpEvent(obj.timeStamp,metadata.timeStamp);
                    metadata.eventStruct.hedTag = eventObj.hedTag;
                    metadata.eventStruct.label = eventObj.label;
                    metadata.eventStruct.latencyInFrame = eventObj.latencyInFrame;
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) obj.numberOfChannels]);
                    
                case 'smooth'
                    prename = 'smooth_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    channels = commandHistory.varargin{3};
                    metadata.numberOfChannels = length(channels);
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                case 'sgolayFilter'
                    prename = 'sgolay_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                case 'filter'
                    prename = 'filt_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    channels = commandHistory.varargin{3};
                    metadata.numberOfChannels = length(channels);
                    metadata.label = obj.label(channels);
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                case 'copyobj'
                    prename = 'copy_';
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    copyfile(obj.binFile,metadata.binFile,'f');
                
                case {'minus' 'plus' 'times' 'rdivide'}
                    op = {'minus' 'plus' 'times' 'rdivide'};
                    I = ismember(op,commandHistory.commandName);
                    prename = [op{I} '_'];
                    metadata.name = [prename metadata.name];
                    metadata.binFile = fullfile(path,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
                    allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
                    
                otherwise
                    newHeader = '';
                    return;
            end
            newHeader = metadata2headerFile(metadata);
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Resample');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'resample',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'filter',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('ICA');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'ica',-1});
            jmenu.add(menuItem);
            %--
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
            menuItem = javax.swing.JMenuItem('Plot spectrum');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrum',-1});
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