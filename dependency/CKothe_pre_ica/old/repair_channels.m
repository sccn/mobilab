function [signal,filled_in] = repair_channels(signal,min_corr,history_fraction,history_len,window_len,prefer_ica,pca_flagquant,pca_maxchannels)
% Repair (interpolate) broken channels.
% [Signal,FilledIn] = repair_channels(Signal,MinCorrelation,HistoryFraction,HistoryLength,WindowLength,PreferICAoverPCA,PCACleanliness,PCAForgiveChannels)
%
% This is an automated artifact rejection function which interpolates channels based on the others
% when they become decorrelated (e.g. loose). For large numbers of channels (128 or more) this function
% is computationally quite heavy as it computes the correlation between each channel and every other.
%
% In:
%   Signal          : continuous data set; note: this data set should be appropriately high-passed,
%                     e.g. using flt_iir.
%
%   MinimumCorrelation  : if a channel has less correlation than this value with any other channel
%                         in a time window, then it is considered "bad" during that window; a channel
%                         gets removed if it is bad for a sufficiently long time period (default: 0.6)
%
%   HistoryFraction : minimum fraction of time (in recent history) during which a channel must have
%                     been flagged as "bad" for it to be interpolated for a given time point. (default: 0.5)
%
%   HistoryLength   : length of channel "badness" history that HistoryFraction refers to, in seconds
%                     If you set this to a higher value (e.g. 30 seconds), you will get fewer spurious
%                     rejections, but some periods of data will be missed (default: 0)
%
%   WindowLength    : length of the windows (in seconds) for which channel "badness" is computed,
%                     i.e. time granularity of the measure; ideally short enough to reasonably
%                     capture periods of artifacts, but no shorter (otherwise the statistic becomes
%                     too noisy) (default: 3)
%
%
%   The following arguments usually don't need to be changed:
% 
%   PreferICAoverPCA : Prefer ICA if available. If you have an ICA decomposition in your data, it 
%                      will be used and no PCA will be computed. If you don''t trust that ICA
%                      decomposition to be good enough set this to false (default: true)
%
%   PCACleanliness   : Rejetion quantile for PCA. If you don''t have a good ICA decomposition for
%                      your data, this is the quantile of data windows that are rejected/ignored
%                      before a PCA correlation matrix is estimated; the higher, the cleaner the PCA
%                      matrix will be (but the less data remains to estimate it). (default: 0.25)
%
%   PCAForgiveChannels : Ignored channel fraction for PCA. If you don''t have a good ICA decomposition 
%                        for your data, if you know that some of your channels are broken
%                        practically in the entire recording, this fraction would need to cover them
%                        (plus some slack). This is the fraction of broken channels that PCA will
%                        accept in the windows for which it computes correlations. The lower this
%                        is, the less data will remain to estimate the correlation matrix but more
%                        channels will be estimated properly. (default: 0.1)
%                      
% Out:
%   Signal : data set with bad channels removed
%
%   FilledIn : matrix of the same size as EEG.data that is 1 at the locations that were filled in and 0
%              otherwise
%
% Examples:
%   % use with defaults
%   eeg = repair_channels(eeg);
%
%   % override the MinimumCorrelation default (making it more aggressive)
%   eeg = repair_channels(eeg,0.7);
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-01-10

if ~exist('min_corr','var') || isempty(min_corr) min_corr = 0.6; end;
if ~exist('history_fraction','var') || isempty(history_fraction) history_fraction = 0.5; end;
if ~exist('history_len','var') || isempty(history_len) history_len = 0; end;
if ~exist('window_len','var') || isempty(window_len) window_len = 3; end;
if ~exist('prefer_ica','var') || isempty(prefer_ica) prefer_ica = true; end;
if ~exist('pca_flagquant','var') || isempty(pca_flagquant) pca_flagquant = 0.25; end;
if ~exist('pca_maxchannels','var') || isempty(pca_maxchannels) pca_maxchannels = 0.1; end;

filled_in = {};
mem_quota = 0.25; % maximum fraction of free memory that may be claimed by this function

% number of data points for our correlation window (N) and recent history (H)
N = window_len * signal.srate; %#ok<*NODEF>
H = history_len * signal.srate;

% first get rid of NaN's
signal.data(isnan(signal.data(:))) = 0;

% make up prior state if necessary
if ~exist('state','var') || isempty(state)
    if size(signal.data,2) < N+1
        error(['The data set needs to be longer than the statistics window length (for this data set ' num2str(window_len) ' seconds).']); end
    % mean, covariance filter conditions, history of "badness" per channel & sample, previously encountered breakage patterns & corresponding reconstruction matrices
    state = struct('ord1',[],'ord2',[],'bad',[],'offset',sum(signal.data,2)/size(signal.data,2),'patterns',[],'matrices',{{}});
    
    if isempty(signal.icawinv) || ~prefer_ica
        % use a generic PCA decomposition
        disp('ChannelRepair: Using a PCA decomposition to interpolate channels; looking for clean data... (note: this may ignore a large fraction of your data -- don''t worry.');
        if exist('clean_windows','file')
            tmp = clean_windows(signal,pca_flagquant,[],pca_maxchannels);
        else
            disp('Note: apparently you are lacking the clean_windows() function -- will calculate the PCA decomposition on the data as-is.');
            tmp = signal;
        end
        disp('done; now repairing... (this may take a while)');
        sphere = 2.0*inv(sqrtm(double(cov(tmp.data')))); %#ok<MINV>
        state.winv = inv(sphere);
        state.chansind = 1:size(signal.data,1);
    else
        disp('ChannelRepair: Using the signal''s ICA decomposition to interpolate channels.');
        % use the ICA decomposition
        state.winv = signal.icawinv;
        state.chansind = signal.icachansind;
    end
    % prepend a made-up data sequence
    signal.data = [repmat(2*signal.data(:,1),1,N) - signal.data(:,(N+1):-1:2) signal.data];
    prepended = true;
else
    prepended = false;
end


% split up the total sample range into k chunks that will fit in memory
[C,S] = size(signal.data);
E = eye(C)~=0;
if S > 1000
    free_memory = java.lang.management.ManagementFactory.getOperatingSystemMXBean().getFreePhysicalMemorySize();
    numsplits = ceil((C^2*S*8*5) / (free_memory*mem_quota));
else
    numsplits = 1;
end
for i=0:numsplits-1
    range = 1+floor(i*S/numsplits) : min(S,floor((i+1)*S/numsplits));
    % get raw data X (-> note: we generally restrict ourselves to channels covered by the decomposition)
    X = double(bsxfun(@minus,signal.data(state.chansind,range),state.offset(state.chansind)));
    % ... and running mean E[X]
    [X_mean,state.ord1] = moving_average(N,X,state.ord1,2);
    % get unfolded cross-terms tensor X*X'
    [m,n] = size(X); X2 = reshape(bsxfun(@times,reshape(X,1,m,n),reshape(X,m,1,n)),m*m,n);
    % ... and running mean of that E[X*X']
    [X2_mean,state.ord2] = moving_average(N,X2,state.ord2,2);
    % compute running covariance E[X*X'] - E[X]*E[X]'
    X_cov = X2_mean - reshape(bsxfun(@times,reshape(X_mean,1,m,n),reshape(X_mean,m,1,n)),m*m,n);
    % get running std dev terms
    X_std = sqrt(X_cov(E,:));
    % clear the diagonal covariance terms
    X_cov(E,:) = 0;
    % cross-multiply std dev terms
    X_crossvar = bsxfun(@times,reshape(X_std,1,m,n),reshape(X_std,m,1,n));
    % normalize the covariance by it (turning it into a running correlation)
    X_corr = X_cov ./ reshape(X_crossvar,m*m,n);
    
    % calculate the per-sample maximum correlation
    X_maxcorr = reshape(max(reshape(X_corr,m,m,n)),m,n);
    % calculate the per-sample 'badness' criterion
    X_bad = X_maxcorr < min_corr;
    
    % filter this using a longer moving average to get the fraction-of-time-bad property
    if history_len > 0
        [X_fracbad,state.bad] = moving_average(H,X_bad,state.bad,2);
        % get the matrix of channels that need to be filled in
        X_fillin = X_fracbad > history_fraction;
    else
        X_fillin = X_bad;
    end
    
    % create a mask of samples that need handling
    X_pattern = sum(X_fillin);
    X_mask = X_pattern>0;
    % get the unique breakage patterns in X_fillin
    [patterns,dummy,occurrence] = unique(X_fillin(:,X_mask)','rows'); %#ok<ASGLU>
    % and the occurrence mask
    X_pattern(X_mask) = occurrence;
    
    % for each pattern...
    for p=1:size(patterns,1)
        patt = patterns(p,:);
        % does it match any known pattern in our state?
        try
            match = all(bsxfun(@eq,patt,state.patterns)');
            reconstruct = state.matrices{match};
        catch
            % no: generate the corresponding reconstruction matrix first
            M_train = state.winv;
            M_trunc = M_train(~patt,:);
            U_trunc = pinv(M_trunc);
            reconstruct = M_train*U_trunc;
            reconstruct = reconstruct(patt,:);
            % append it to the state's DB
            state.patterns = [state.patterns; patt];
            state.matrices{size(state.patterns,1)} = reconstruct;
        end
        % now reconstruct the corresponding broken channels
        mask = p==X_pattern;
        signal.data(state.chansind(patt),range(mask)) = reconstruct * signal.data(state.chansind(~patt),range(mask));
    end
    
    % append the fill-in matrix
    filled_in{end+1} = X_fillin;
end

% trim the prepended part if there was one
if prepended
    signal.data(:,1:N) = []; end

% and assign the filled_in matrix
filled_in = [filled_in{:}];



function [X,Zf] = moving_average(N,X,Zi,dim)
% Like filter() for the special case of moving-average kernels.
% [X,Zf] = moving_average(N,X,Zi,Dim)
%
% In:
%   N : filter length in samples
%
%   X : data matrix
%
%   Zi : initial filter conditions (default: [])
%
%   Dim : dimension along which to filter (default: first non-singleton dimension)
%
% Out:
%   X : the filtered data
%
%   Zf : final filter conditions
%
% See also:
%   filter

% determine the dimension along which to filter
if nargin <= 3
    if isscalar(X)
        dim = 1;
    else
        dim = find(size(X)~=1,1); 
    end
end

% empty initial state
if nargin <= 2
    Zi = []; end

lenx = size(X,dim);
if lenx == 0
    % empty X
    Zf = Zi;
else
    if N < 100
        % small N: use filter
        [X,Zf] = filter(ones(N,1)/N,1,X,Zi,dim);
    else
        % we try to avoid permuting dimensions below as this would increase the running time by ~3x
        if ndims(X) == 2
            if dim == 1
                % --- process along 1st dimension ---
                if isempty(Zi)
                    % zero initial state
                    Zi = zeros(N,size(X,2));
                elseif size(Zi,1) == N-1
                    % reverse engineer filter's initial state (assuming a moving average)
                    tmp = diff(Zi(end:-1:1,:),1,1);
                    Zi = [tmp(end:-1:1,:); Zi(end,:)]*N;
                    Zi = [-sum(Zi,1); Zi];
                elseif ~isequal(size(Zi),[N,size(X,2)])
                    error('These initial conditions do not have the correct format.');
                end
                
                % pre-pend initial state & get dimensions
                Y = [Zi; X]; M = size(Y,1);
                % get alternating index vector (for additions & subtractions)
                I = [1:M-N; 1+N:M];
                % get sign vector (also alternating, and includes the scaling)
                S = [-ones(1,M-N); ones(1,M-N)]/N;
                % run moving average
                X = cumsum(bsxfun(@times,Y(I(:),:),S(:)),1);
                % read out result
                X = X(2:2:end,:);
                
                % construct final state
                if nargout > 1
                    Zf = [-(X(end,:)*N-Y(end-N+1,:)); Y(end-N+2:end,:)]; end
            else
                % --- process along 2nd dimension ---
                if isempty(Zi)
                    % zero initial state
                    Zi = zeros(N,size(X,1));
                elseif size(Zi,1) == N-1
                    % reverse engineer filter's initial state (assuming a moving average)
                    tmp = diff(Zi(end:-1:1,:),1,1);
                    Zi = [tmp(end:-1:1,:); Zi(end,:)]*N;
                    Zi = [-sum(Zi,1); Zi];
                elseif ~isequal(size(Zi),[N,size(X,1)])
                    error('These initial conditions do not have the correct format.');
                end
                
                % pre-pend initial state & get dimensions
                Y = [Zi' X]; M = size(Y,2);
                % get alternating index vector (for additions & subtractions)
                I = [1:M-N; 1+N:M];
                % get sign vector (also alternating, and includes the scaling)
                S = [-ones(1,M-N); ones(1,M-N)]/N;
                % run moving average
                X = cumsum(bsxfun(@times,Y(:,I(:)),S(:)'),2);
                % read out result
                X = X(:,2:2:end);
                
                % construct final state
                if nargout > 1
                    Zf = [-(X(:,end)*N-Y(:,end-N+1)) Y(:,end-N+2:end)]'; end
            end
        else
            % --- ND array ---
            [X,nshifts] = shiftdim(X,dim-1);
            shape = size(X); X = reshape(X,size(X,1),[]);
            
            if isempty(Zi)
                % zero initial state
                Zi = zeros(N,size(X,2));
            elseif size(Zi,1) == N-1
                % reverse engineer filter's initial state (assuming a moving average)
                tmp = diff(Zi(end:-1:1,:),1,1);
                Zi = [tmp(end:-1:1,:); Zi(end,:)]*N;
                Zi = [-sum(Zi,1); Zi];
            elseif ~isequal(size(Zi),[N,size(X,2)])
                error('These initial conditions do not have the correct format.');
            end
            
            % pre-pend initial state & get dimensions
            Y = [Zi; X]; M = size(Y,1);
            % get alternating index vector (for additions & subtractions)
            I = [1:M-N; 1+N:M];
            % get sign vector (also alternating, and includes the scaling)
            S = [-ones(1,M-N); ones(1,M-N)]/N;
            % run moving average
            X = cumsum(bsxfun(@times,Y(I(:),:),S(:)),1);
            % read out result
            X = X(2:2:end,:);
            
            % construct final state
            if nargout > 1
                Zf = [-(X(end,:)*N-Y(end-N+1,:)); Y(end-N+2:end,:)]; end
            
            X = reshape(X,shape);
            X = shiftdim(X,ndims(X)-nshifts);
        end
    end
end
