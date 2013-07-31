function signal = clean_channels(signal,min_corr,ignored_quantile,window_len,ignored_time,rereferenced)
% Remove channels with abnormal data from a continuous data set. Currently offline only.
% Signal = clean_channels(Signal,MinCorrelation,IgnoredQuantile,WindowLength,MaxIgnoredTime,Rereferenced)
%
% This is an automated artifact rejection function which ensures that the data set contains no
% channels that record complete trash. If channels with control signals are contained in the data,
% these are usually also removed.
%
% In:
%   Signal          : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 2.0Hz transition band)
%
%   MinCorrelation  : minimum correlation between a channel and any other channel (in a window)
%                     below which the window is considered abnormal (default: 0.6)
%                     
%
%   IgnoredQuantile : quantile of the (sorted) correlation values that is ignored, including
%                     self-correlation (default: 0.1)
%
%   WindowLength    : length of the windows (in seconds) for which correlation is computed; ideally
%                     short enough to reasonably capture periods of global artifacts (which are
%                     ignored), but no shorter (for computational reasons) (default: 1)
% 
%   MaxIgnoredTime  : maximum time (in seconds) in the data set that is ignored (can contain
%                     arbitrary data without affecting the outcome) may also be a fraction of the total
%                     data set length (default: 0.4)
%
%   Rereferenced    : whether the operation should be run on re-referenced data (default: false)
%
% Out:
%   Signal : data set with bad channels removed
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

if ~exist('min_corr','var') || isempty(min_corr) min_corr = 0.6; end
if ~exist('ignored_quantile','var') || isempty(ignored_quantile) ignored_quantile = 0.1; end
if ~exist('window_len','var') || isempty(window_len) window_len = 1; end
if ~exist('ignored_time','var') || isempty(ignored_time) ignored_time = 0.4; end
if ~exist('rereferenced','var') || isempty(rereferenced) rereferenced = false; end

% flag channels
if ~exist('removed_channels','var')
    if ignored_time > 0 && ignored_time < 1  %#ok<*NODEF>
        ignored_time = size(signal.data,2)*ignored_time;
    else
        ignored_time = signal.srate*ignored_time;
    end
    
    [C,S] = size(signal.data);
    window_len = window_len*signal.srate;
    wnd = 0:window_len-1;
    offsets = 1:window_len:S-window_len;
    W = length(offsets);    
    retained = 1:(C-ceil(C*ignored_quantile));

    % optionally subtract common reference from data
    if rereferenced
        X = signal.data - repmat(mean(signal.data),C,1);
    else
        X = signal.data;
    end
    
    flagged = zeros(C,W);
    % for each window, flag channels with too low correlation to any other channel (outside the
    % ignored quantile)
    for o=1:W
        sortcc = sort(abs(corrcoef(X(:,offsets(o)+wnd)')));
        flagged(:,o) = all(sortcc(retained,:) < min_corr);
    end
    % mark all channels for removal which have more flagged samples than the maximum number of
    % ignored samples
    removed_channels = find(sum(flagged,2)*window_len > ignored_time);
end

% execute
signal = pop_select(signal,'nochannel',removed_channels);

