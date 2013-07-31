function signal = clean_flatlines(signal,min_stddev)
% Remove (near-) flat-lined channels.
% Signal = clean_flatlines(Signal,MinimumStdDev)
%
% This is an automated artifact rejection function which ensures that 
% the data contains no flat-lined channels.
%
% In:
%   Signal          : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 2.0Hz transition band)
%
%   MinimumStdDev : The minimum tolerated channel std-dev (in fact median absolute deviation).
%                   If a channel has a lower standard-deviation than this, it will be considered abnormal.
%                   (default: 0.001)
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

if ~exist('min_stddev','var') || isempty(min_stddev) min_stddev = 0.001; end

% flag channels
removed_channels = mad(double(signal.data),1,2) < min_stddev;

% execute
if ~isempty(removed_channels)
    disp('Now removing flat-line channels...');
    signal = pop_select(signal,'nochannel',find(removed_channels)); 
    if isfield(signal.etc,'clean_channel_mask')
        signal.etc.clean_channel_mask(signal.etc.clean_channel_mask) = ~removed_channels;
    else
        signal.etc.clean_channel_mask = ~removed_channels;
    end
end
