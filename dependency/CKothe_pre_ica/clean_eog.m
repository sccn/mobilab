function [signal,state] = clean_eog(signal,eogchans,ffact,kernellen,removeeog,state)
% Remove EOG artifacts from EEG using EOG reference channels.
% [Signal,State] = clean_eog(Signal, EOGChannels, ForgetFactor, KernelLength,RemoveEOG,State)
%
% This is an online filter that operates on continuous data, and removes EOG using a regression
% technique, if EOG channels are present (using recursive least squares) [1]. Note that noise in
% the EOG signals may be transferred onto the EEG channels.
%
% In:
%   Signal       :   continuous data set to be filtered
%
%   EOGChannels  :   list of EOG channel indices or cell array of EOG channel names
%	             (if empty: try to auto-detect)
%
%   ForgetFactor :   forgetting factor of the adaptive filter; amounts to a choice of the 
%                    effective memory length (default: 0.9995)
%
%   KernelLength ;   length/order of the temporal FIR filter kernel (default: 3)
%
%   RemoveEOG    :   whether to remove the EOG channels after processing (default: false)
%
%   State        :   previous filter state, as obtained by a previous execution of flt_eog on an
%                    immediately preceding data set (default: [])
%
% Out:
%   Signal       :  filtered, continuous EEGLAB data set
%
%   State        :  state of the filter, can be used to continue on a subsequent portion of the data
%
% References:
%  [1] P. He, G.F. Wilson, C. Russel, "Removal of ocular artifacts from electro-encephalogram by adaptive filtering"
%      Med. Biol. Eng. Comput. 42 pp. 407-412, 2004
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-04-17

if ~exist('eogchans','var') eogchans = []; end
if ~exist('ffact','var') || isempty(ffact) ffact = 0.9995; end
if ~exist('kernellen','var') || isempty(kernellen) kernellen = 3; end
if ~exist('removeeog','var') || isempty(removeeog) removeeog = false; end

if size(signal.data,3) > 1
    error('flt_eog is supposed to be applied to continuous (non-epoched) data.'); end

% initialize the state, if necessary
if ~exist('state','var')
    % figure out what the EOG channels are
    if iscell(eogchans)
        [x,a,b] = intersect(eogchans,{signal.chanlocs.labels}); %#ok<ASGLU>
        [x,I] = sort(a); eogchans = b(I); %#ok<ASGLU>
    elseif isempty(eogchans)
        eogchans = find(strncmpi('eo',{signal.chanlocs.labels},2) | ~cellfun('isempty',strfind(lower({signal.chanlocs.labels}),'eog')));
    end
    if isempty(eogchans)
        error('Could not find EOG channels in the data; please specify the names / indices of EOG channels explicitly.'); end
    state.eog = eogchans;                          % eog channel indices
    state.eeg = setdiff(1:signal.nbchan,eogchans); % eeg channel indices
    state.neog = length(state.eog);                % number of eog channel indices
    
    % initialize RLS filter state
    state.hist = zeros(state.neog,kernellen);     % hist is the block of the M last eog samples in matrix form
    state.R_n = eye(state.neog * kernellen) / 0.01; % R(n-1)^-1 is the inverse matrix
    state.H_n = zeros(state.neog*kernellen,length(state.eeg));  % H(n-1) is the EOG filter kernel
end

% apply filter
[X,state.hist,state.H_n,state.R_n] = compute(signal.data,state.hist,state.H_n,state.R_n,state.eeg,state.eog,ffact);

if removeeog
    % Note: the proper way would be to use pop_select...
    signal.data = X(state.eeg,:);
    signal.nbchan = size(signal.data,1);
    signal.chanlocs = signal.chanlocs(signal.eeg);
else
    signal.data = X;
end



function [X,hist,H_n,R_n] = compute(X,hist,H_n,R_n,eeg,eog,ffact)
% for each sample...
for n=1:size(X,2)
    % update the EOG history by feeding in a new sample
    hist = [hist(:,2:end) X(eog,n)];
    % vectorize the EOG history into r(n)        % Eq. 23
    tmp = hist';
    r_n = tmp(:);
    
    % calculate K(n)                             % Eq. 25
    K_n = R_n * r_n / (ffact + r_n' * R_n * r_n);
    % update R(n)                                % Eq. 24
    R_n = ffact^-1 * R_n - ffact^-1 * K_n * r_n' * R_n;
    
    % get the current EEG samples s(n)
    s_n = X(eeg,n);    
    % calculate e(n/n-1)                         % Eq. 27
    e_nn = s_n - (r_n' * H_n)';    
    % update H(n)                                % Eq. 26
    H_n = H_n + K_n * e_nn';
    % calculate e(n), new cleaned EEG signal     % Eq. 29
    e_n = s_n - (r_n' * H_n)';
    % write back into the signal
    X(eeg,n) = e_n;
end
