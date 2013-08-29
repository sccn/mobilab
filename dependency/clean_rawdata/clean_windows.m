function [signal,sample_mask] = clean_windows(signal,max_bad_channels,zthresholds,window_len,censor_cutoffs,censor_dropouts,censor_dropout_maxreject)
% Remove periods with abnormally high-power content from continuous data.
% [Signal,Mask] = clean_windows(Signal,MaxBadChannels,BadTolerances,)
%
% This function cuts segments from the data which contain high-power artifacts. Specifically,
% only windows are retained which have less than a certain fraction of "bad" channels, where a channel
% is bad in a window if its power is above or below a given upper/lower threshold (in standard 
% deviations from a robust estimate of the EEG power distribution in the channel).
%
% In:
%   Signal         : Continuous data set, assumed to be appropriately high-passed (e.g. >1Hz or
%                    0.5Hz - 2.0Hz transition band)
%
%   MaxBadChannels : The maximum number or fraction of bad channels that a retained window may still
%                    contain (more than this and it is removed). Reasonable range is 0.05 (very clean
%                    output) to 0.3 (very lax cleaning of only coarse artifacts). Default: 0.1.
%
%   PowerTolerances: The minimum and maximum standard deviations within which the power of a channel
%                    must lie (relative to a robust estimate of the clean EEG power distribution in 
%                    the channel) for it to be considered "not bad". Default: [-5 3.5].
%
%
%   The following are "specialty" parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these to your data.
%
%   WindowLength    : Window length that is used to check the data for artifact content. This is 
%                     ideally as long as the expected time scale of the artifacts but not shorter 
%                     than half a cycle of the high-pass filter that was used. Default: 1.
% 
%   CensorCutoffs : These are robust data censoring cutoffs, in robust z scores per channel, for
%                   estimating the EEG power distribution per channel. This does not need to be
%                   tuned unless the artifact content is very unusual, e.g. lots of flat channel
%                   drop-outs or a very uncommon type of noise. Default: [-5 4].
%
%   CensorDropouts : This is an extra censoring cutoff, in robust z scores across all channels,
%                    to handle data where the majority of samples in a channel (but not all) drop out.
%                    Default: -3.
%
%   CensorDropoutMaxReject : This is the maximum fraction of windows rejected for which the dropout
%                            censoring is considered applicable (otherwise the criterion is ignored,
%                            e.g. for very low-amplitude channels). Default: 0.75
%
% Out:
%   Signal : data set with bad time periods removed.
%
%   Mask   : mask of retained samples (logical array)
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

warning off stats:gevfit:IterLimit
warning off stats:gamfit:IterLimit

if ~exist('max_bad_channels','var') || isempty(max_bad_channels) max_bad_channels = 0.15; end
if ~exist('zthresholds','var') || isempty(zthresholds) zthresholds = [-5 3.5]; end
if ~exist('window_len','var') || isempty(window_len) window_len = 1; end
if ~exist('censor_cutoffs','var') || isempty(censor_cutoffs) censor_cutoffs = [-5 4]; end
if ~exist('censor_dropouts','var') || isempty(censor_dropouts) censor_dropouts = -3; end
if ~exist('censor_dropout_maxreject','var') || isempty(censor_dropout_maxreject) censor_dropout_maxreject = 0.75; end

[C,S] = size(signal.data);
N = window_len*signal.srate;
wnd = 0:N-1;
offsets = round(1:N/2:S-N);
W = length(offsets);

if max_bad_channels < 1
    max_bad_channels = round(max_bad_channels*C); end
if max_bad_channels >= C
    sample_mask = true(1,S);
    return; 
end

% compute windowed power across all channels
wp = zeros(C,W);
for c = 1:C
    for o=1:W
        x = signal.data(c,offsets(o) + wnd);
        wp(c,o) = sqrt(sum(x.*x)/N); 
    end
end

% find the median and median absolute deviation for that
mu_all = median(wp(:));
st_all = mad(wp(:),1);

wz = zeros(C,W);
for c = 1:C
    tmp = wp(c,:);
    % censor values that are likely temporary channel drop-outs using information from all channels
    tmp = tmp(tmp>(mu_all+st_all*censor_dropouts));
    if length(tmp)/W < (1-censor_dropout_maxreject)
        % if some channel is extremely low-amplitude, we cannot apply this criterion
        tmp = wp(c,:); 
    end
    % estimate the robust mean and std. deviation for this channel
    mu_robust = median(tmp);
    st_robust = mad(tmp,1);
    % censor values that are extreme outliers for this particular channel
    tmp = tmp(tmp<(mu_robust+st_robust*censor_cutoffs(2)) & tmp>(mu_robust+st_robust*censor_cutoffs(1)));
    % fit a generalized extreme value distribution and a gamma distribution and compute the modes
    params_gam = gamfit(tmp); 
    mode_gam = (params_gam(1) - 1)*params_gam(2);
    params_gev = gevfit(tmp);
    if params_gev(1) ~= 0
        mode_gev = params_gev(3) + params_gev(2)*((1+params_gev(1))^-params_gev(1) - 1)/params_gev(1);
    else
        mode_gev = params_gev(3);
    end
    % we take the smaller mode as the mean of the underlying EEG distribution (assuming that it is a Gaussian component of that distribution)
    % this is because the GEV allows right-skewed fits, which are non-physiological (in this case we fall back to the Gamma fit)
    mu_fine = min([mode_gam,mode_gev]);
    % we estimate the standard deviation from the absolute deviation from the mode, using only
    % data smaller than the mode
    st_fine = median(abs(tmp(tmp<mu_fine)-mu_fine))*1.3652;
    % calculate the channel power in z scores relative to the estimated underlying EEG distribution
    wz(c,:) = (wp(c,:) - mu_fine)/st_fine;
end

% sort z scores into quantiles
swz = sort(wz);

% determine which windows to retain
retained_mask = true(1,W);
if max(zthresholds)>0
    retained_mask(swz(end-max_bad_channels,:) > max(zthresholds)) = false; end
if min(zthresholds)<0
    retained_mask(swz(1+max_bad_channels,:) < min(zthresholds)) = false; end
retained_windows = find(retained_mask);

% find retained samples
retained_samples = repmat(offsets(retained_windows)',1,length(wnd))+repmat(wnd,length(retained_windows),1);

% mask them out
sample_mask = false(1,S); sample_mask(retained_samples(:)) = true;
fprintf('Keeping %.1f%% (%.0f seconds) of the data.\n',100*(mean(sample_mask)),nnz(sample_mask)/signal.srate);
% collect into intervals
retain_data_intervals = reshape(find(diff([false sample_mask false])),2,[])';
retain_data_intervals(:,2) = retain_data_intervals(:,2)-1;

% apply selection
signal = pop_select(signal, 'point', retain_data_intervals);

if isfield(signal.etc,'clean_sample_mask')
    signal.etc.clean_sample_mask(signal.etc.clean_sample_mask) = sample_mask;
else
    signal.etc.clean_sample_mask = sample_mask;
end
