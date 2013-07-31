function EEG = clean_artifacts(EEG,channel_crit,burst_crit,window_crit,highpass,channel_crit_excluded,channel_crit_maxbad,burst_crit_refsect,burst_crit_refwndlen,burst_flt_settings,flatline_crit)
% Clean various types of artifacts from the EEG/
% EEG = clean_artifacts(EEG,Highpass,ChannelCriterion,BurstCriterion,WindowCriterion,ChannelCriterionExcluded,ChannelCriterionMaxBadTime,BurstCriterionRefSection)
%
% This function removes drifts, bad channels, short-time bursts, and bad time windows from the data.
% Tip: Any parameter can also be passed in as [] to use the respective default or as 'off' to disable it.
%
% In:
%   EEG : Raw continuous EEG recording to clean up.
%
%
%   NOTE: The following parameters are the core parameters of the cleaning procedure. If the method
%   removes too many (or too few) channels, time windows, or burst artifacts, you need to tune these
%   values. Hopefully you only need to do this in rare cases.
%
%   ChannelCriterion : Criterion for removing bad channels. This is a minimum correlation
%                      value that a given channel must have w.r.t. at least one other channel.
%                      Generally, a fraction of most correlated channels is excluded from this 
%                      measure. A higher value makes the criterion more aggressive. Default: 0.55.
%                      Reasonable range: 0.45 (very lax) - 0.65 (quite aggressive).
%
%   BurstCriterion : Criterion for projecting local bursts out of the data. This is the standard
%                    deviation from clean EEG at which a signal component would be considered a
%                    burst artifact. Generally a higher value makes the criterion less aggressive.
%                    Default: 3. Reasonable range: 2.5 (very aggressive) to 5 (very lax). One
%                    usually does not need to tune this parameter.
%
%   WindowCriterion : Criterion for removing bad time windows. This is a quantile of the per-
%                     window variance that should be considered for removal. Multiple channels
%                     need to be in that quantile for the window to be removed. Generally a
%                     higher value makes the criterion more aggressive. Default: 0.1. Reasonable
%                     range: 0.05 (very lax) to 0.15 (quite aggressive).
%
%   Highpass : Transition band for the initial high-pass filter in Hz (default [0.5 1]).
%              This is [transition-start, transition-end].
%
%
%   NOTE: The following are "specialty" parameters that may be tuned if one of the criteria does
%   not seem to be doing the right thing. These basically amount to side assumptions about the
%   data that usually do not change much across recordings, but sometimes do.
%
%   ChannelCriterionExcluded : The fraction of excluded most correlated channels when computing the
%                              Channel criterion. This adds robustness against channels that are
%                              disconnected and record the same noise process. At most this fraction
%                              of all channels may be fully disconnected. Default: 0.15. Reasonable
%                              range: 0.1 (fairly lax) to 0.3 (very aggressive); note that
%                              increasing this value requires the ChannelCriterion to be relaxed to
%                              maintain the same overall amount of removed channels.
%
%   ChannelCriterionMaxBadTime : This is the maximum fraction of the data set during which a channel
%                                may be bad without being considered "bad". Generally a lower value 
%                                makes the criterion more aggresive. Default: 0.4. Reasonable range:
%                                0.15 (very aggressive) to 0.5 (very lax).
%
%   BurstCriterionRefSection : The quantile of the data that is used as clean reference EEG. Instead of 
%                              a number one may also directly pass in a data set that contains clean 
%                              reference data (for example a minute of resting EEG). A lower values makes this
%                              criterion more aggressive. Default: as in repair_bursts (0.66). Reasonable range: 
%                              0.5 (very aggressive) to 0.75 (quite lax). Inside note: the actual rule is a 
%                              bit more complicated since a window must have a certain fraction of channels
%                              in this quantile to be considered part of the bad data.
%
%   BurstCriterionRefWindowLen : Length of the windows that are considered reference-quality EEG. This 
%                                should be on a time scale of multiple seconds and match the duration
%                                of what can be expected a good clean-data period. Default: As in 
%                                repair_bursts (7)
%
%   BurstFilterSettings : Parameters for the spectrum-shaping filter of the burst criterion that 
%                         reweights frequencies according to their relevance for artifact detection. 
%                         This is a cell array of {Frequencies, Amplitudes, FilterOrder},
%                         see also repair_bursts. Default: as in repair_bursts.
%
%
%   FlatlineCriterion : Minimum standard-deviation that a channel must have to be considered valid.
%                       Default: 0.0005
%
%
% Out:
%   EEG : Cleaned EEG recording.
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-09-04

if ~exist('highpass','var') || isempty(highpass) highpass = [0.5 1]; end
if ~exist('channel_crit','var') || isempty(channel_crit) channel_crit = 0.45; end
if ~exist('burst_crit','var') || isempty(burst_crit) burst_crit = 3; end
if ~exist('window_crit','var') || isempty(window_crit) window_crit = 0.1; end
if ~exist('channel_crit_excluded','var') || isempty(channel_crit_excluded) channel_crit_excluded = 0.1; end
if ~exist('channel_crit_maxbad','var') || isempty(channel_crit_maxbad) channel_crit_maxbad = 0.5; end
if ~exist('burst_crit_refsect','var') || isempty(burst_crit_refsect) burst_crit_refsect = []; end
if ~exist('burst_crit_refwndlen','var') || isempty(burst_crit_refwndlen) burst_crit_refwndlen = []; end
if ~exist('burst_flt_settings','var') || isempty(burst_flt_settings) burst_flt_settings = []; end
if ~exist('flatline_crit','var') || isempty(flatline_crit) flatline_crit = 0.001; end

% remove flat-line channels
if ~strcmp(flatline_crit,'off')
    EEG = clean_flatlines(EEG,flatline_crit); end

% high-pass filter the data
if ~strcmp(highpass,'off')
    EEG = clean_drifts(EEG,highpass); end

% remove hopeless channels
if ~strcmp(channel_crit,'off')
    EEG = clean_channels(EEG,channel_crit,channel_crit_excluded,[],channel_crit_maxbad); end

% repair bursts
if ~strcmp(burst_crit,'off')
    EEG = repair_bursts(EEG,burst_crit,[],[],[],burst_crit_refsect,[],burst_crit_refwndlen,burst_flt_settings); end

% remove hopeless time windows
if ~strcmp(window_crit,'off')
    EEG = clean_windows(EEG,window_crit); end

