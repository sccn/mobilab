function [signal,sample_mask] = clean_windows(signal,flag_quantile,window_len,min_badchans)
% Remove periods of abnormal data from continuous data.
% [Signal,Mask] = clean_windows(Signal,FlaggedQuantile,WindowLength,MaxIgnoredChannels)
%
% This is an autmated artifact rejection function which cuts segments of artifacts (characterized by
% their signal power) from the data.
%
% In:
%   Signal          : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     0.5Hz - 2.0Hz transition band)
%
%   FlaggedQuantile : upper quantile of the per-channel windows that should be flagged for potential
%                     removal (removed if flagged in all except for some possibly bad channels);
%                     controls the aggressiveness of the rejection; if two numbers are specified,
%                     the first is the lower quantile and the second is the upper quantile to be
%                     removed (default: 0.15)
%
%   WindowLength    : length of the windows (in seconds) which are inspected for artifact content;
%                     ideally as long as the expected time scale of the artifacts (e.g. chewing)
%                     (default: 1)
% 
%   MinAffectedChannels : if for a time window more than this number (or ratio) of channels are
%                         affected (i.e. flagged), the window will be considered "bad". (default: 0.5)
%
% Out:
%   Signal : data set with bad time periods (and all events) removed, if keep_metadata is false.
%
%   Mask   : mask of retained samples (logical array)
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

if ~exist('flag_quantile','var') || isempty(flag_quantile) flag_quantile = 0.15; end
if ~exist('window_len','var') || isempty(window_len) window_len = 1; end
if ~exist('min_badchans','var') || isempty(min_badchans) min_badchans = 0.5; end

if ~isempty(min_badchans) && min_badchans > 0 && min_badchans < 1 %#ok<*NODEF>
    min_badchans = size(signal.data,1)*min_badchans; end
if isscalar(flag_quantile)
    flag_quantile = [0 flag_quantile]; end

[C,S] = size(signal.data);
window_len = window_len*signal.srate;
wnd = 0:window_len-1;
offsets = round(1:window_len/2:S-window_len);
W = length(offsets);

wpwr = zeros(C,W);
% for each channel
for c = 1:C
    % for each window
    for o=1:W
        % extract data
        x = signal.data(c,offsets(o) + wnd);
        % compute windowed power (measures both mean deviations, i.e. jumps, and large oscillations)
        wpwr(c,o) = sqrt(sum(x.^2)/window_len);
    end
end

[dummy,i] = sort(wpwr'); %#ok<TRSRT,ASGLU>

% find retained window indices per channel
retained_quantiles = i(1+floor(W*flag_quantile(1)):round(W*(1-flag_quantile(2))),:)';

% flag them in a Channels x Windows matrix (this can be neatly visualized)
retain_mask = zeros(C,W);
for c = 1:C
    retain_mask(c,retained_quantiles(c,:)) = 1; end

% find retained windows
retained_windows = find(sum(1-retain_mask) <= min_badchans);
% find retained samples
retained_samples = repmat(offsets(retained_windows)',1,length(wnd))+repmat(wnd,length(retained_windows),1);
% mask them out
sample_mask = false(1,S); sample_mask(retained_samples(:)) = true;
fprintf('Removing %.1f%% (%.0f seconds) of the data.\n',100*(1-mean(sample_mask)),nnz(~sample_mask)/signal.srate);

% retain the masked data, update meta-data appropriately
retain_data_intervals = reshape(find(diff([false sample_mask false])),2,[])';
retain_data_intervals(:,2) = retain_data_intervals(:,2)-1;
signal = pop_select(signal, 'point', retain_data_intervals);

