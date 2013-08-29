function signal = clean_flatlines(signal,max_flatline_duration,max_allowed_jitter)
% Remove (near-) flat-lined channels.
% Signal = clean_flatlines(Signal,MinimumStdDev)
%
% This is an automated artifact rejection function which ensures that 
% the data contains no flat-lined channels.
%
% In:
%   Signal : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%            with a 0.5Hz - 2.0Hz transition band)
%
%   MaxFlatlineDuration : Maximum tolerated flatline duration. In seconds. If a channel has a longer
%                         flatline than this, it will be considered abnormal. Default: 5
%
%   MaxAllowedJitter : Maximum tolerated jitter during flatlines. As a multiple of epsilon.
%                      Default: 20
%
% Out:
%   Signal : data set with flat channels removed
%
% Examples:
%   % use with defaults
%   eeg = clean_flatlines(eeg);
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-08-30

if ~exist('max_flatline_duration','var') || isempty(max_flatline_duration) max_flatline_duration = 5; end
if ~exist('max_allowed_jitter','var') || isempty(max_allowed_jitter) max_allowed_jitter = 20; end

% flag channels
removed_channels = [];
for c=1:signal.nbchan
    zero_intervals = reshape(find(diff([false abs(diff(signal.data(c,:)))<(max_allowed_jitter*eps) false])),2,[])';
    if max(zero_intervals(:,2) - zero_intervals(:,1)) > max_flatline_duration*signal.srate
        removed_channels(end+1) = c; end
end

% remove them
if ~isempty(removed_channels)
    disp('Now removing flat-line channels...');
    signal = pop_select(signal,'nochannel',find(removed_channels)); 
    if isfield(signal.etc,'clean_channel_mask')
        signal.etc.clean_channel_mask(signal.etc.clean_channel_mask) = ~removed_channels;
    else
        signal.etc.clean_channel_mask = ~removed_channels;
    end
end
