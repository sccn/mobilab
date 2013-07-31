function [EEG,HP,BUR] = clean_artifacts(EEG,varargin)
% Clean various types of artifacts from the EEG.
% [EEG,HP,BUR] = clean_artifacts(EEG, Options...)
%
% This function removes drifts, bad channels, generic short-time bursts and bad segments from the data.
% Tip: Any parameter can also be passed in as [] to use the respective default or as 'off' to disable it.
% 
% Hopefully parameter tuning should be the exception when using this function -- the only parameter
% that likely requires a setting is the BurstCriterion. For a clean ERP experiment with little room
% for subject movement the recommended setting is 4. For movement experiments or otherwise noisy
% recordings the default setting of 3 is okay. See also examples at the bottom of the help.
%
% In:
%   EEG : Raw continuous EEG recording to clean up.
%
%
%   NOTE: The following parameters are the core parameters of the cleaning procedure; they are
%   passed in as Name-Value Pairs. If the method removes too many (or too few) channels, time
%   windows, or burst artifacts, you need to tune these values. Hopefully you only need to do this
%   in rare cases.
%
%   ChannelCriterion : Criterion for removing bad channels. This is a minimum correlation
%                      value that a given channel must have w.r.t. a fraction of other channels. A
%                      higher value makes the criterion more aggressive. Reasonable range: 0.4 (very
%                      lax) - 0.6 (quite aggressive). Default: 0.45.
%
%   BurstCriterion : Criterion for projecting local bursts out of the data. This is in standard
%                    deviations from clean EEG at which a signal component would be considered a
%                    burst artifact. Generally a lower value makes the criterion more aggressive.
%                    Reasonable range: 2.5 (very aggressive, cuts some EEG) to 5 (very lax, cuts
%                    almost never EEG). Default: 3.
%
%   WindowCriterion : Criterion for removing bad time windows. This is the maximum fraction of bad
%                     channels that are tolerated in the final output data for each considered window.
%                     Generally a lower value makes the criterion more aggressive. Default: 0.05.
%                     Reasonable range: 0.05 (very aggressive) to 0.3 (very lax).
%
%   Highpass : Transition band for the initial high-pass filter in Hz. This is formatted as
%              [transition-start, transition-end]. Default: [0.5 1].
%
%
%   NOTE: The following are "specialty" parameters that may be tuned if one of the criteria does
%   not seem to be doing the right thing. These basically amount to side assumptions about the
%   data that usually do not change much across recordings, but sometimes do.
%
%   ChannelCriterionExcluded : The fraction of channels that must have at least the given correlation 
%                              value when computing the Channel criterion. This adds robustness
%                              against pairs of channels that are shorted or other that are
%                              disconnected but record the same noise process. Reasonable range: 0.1
%                              (fairly lax) to 0.3 (very aggressive); note that increasing this
%                              value requires the ChannelCriterion to be relaxed in order to
%                              maintain the same overall amount of removed channels. Default: 0.1.
%
%   ChannelCriterionMaxBadTime : This is the maximum fraction of the recording during which a channel
%                                may be bad without being removed. Generally a lower value makes the
%                                criterion more aggresive. Reasonable range: 0.15 (very aggressive)
%                                to 0.6 (very lax). Default: 0.5.
%
%   BurstCriterionRefMaxBadChns: The maximum fraction of bad channels per time window of the data that
%                                is used as clean reference EEG for the burst criterion. Instead of
%                                a number one may also directly pass in a data set that contains clean
%                                reference data (for example a minute of resting EEG). A lower value
%                                makes this criterion more aggressive. Reasonable range: 0.05 (very
%                                aggressive) to 0.3 (quite lax). If you have lots of little glitches
%                                in a few channels that don't get entirely cleaned you might want to 
%                                reduce this number so that they don't go into the calibration data. 
%                                Default: 0.075.
%
%   BurstCriterionRefTolerances : These are the power tolerances beyond which a channel in the clean
%                                 reference data is considered "bad", in standard deviations relative
%                                 to a robust EEG power distribution (lower and upper bound).
%                                 Default: [-5 3.5].
%
%   BurstFilterSettings : Parameters for the spectral filter used in the burst criterion that determines
%                         the weighting of frequencies in the calculation of statistics. This is a cell
%                         array of {Frequencies, Amplitudes, FilterOrder}, see also repair_bursts.
%                         Can also be either 'basic' or 'sensitive'. Default: like in repair_bursts. 
%
%   WindowCriterionTolerances : These are the power tolerances beyond which a channel in the final
%                               output data is considered "bad", in standard deviations relative
%                               to a robust EEG power distribution (lower and upper bound).
%                               Default: [-5 5].
%
%   FlatlineCriterion : Minimum standard-deviation that a channel must have to be considered valid.
%                       Default: 0.001
%
% Out:
%   EEG : Final cleaned EEG recording.
%
%   HP : Also just the original high-pass filtered recording.
%
%   BUR : Also just the burst-cleaned recording (with no windows removed).
%
% Examples:
%   % Load a recording, clean it, and visualize the difference (using the defaults)
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw);
%   vis_artifacts(clean,raw);
%
%   % Instead using the setting for clean ERP tasks instead:
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw,[],4);
%   vis_artifacts(clean,raw);
%
%   % Passing some parameter by name (here making the WindowCriterion setting less picky)
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw,'WindowCriterion',0.25);
%
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-09-04

parse_arguments(varargin,...
    {'channel_crit','ChannelCriterion'}, 0.45, ...
    {'burst_crit','BurstCriterion'}, 3, ...
    {'window_crit','WindowCriterion'}, 0.1, ...
    {'highpass','Highpass'}, [0.5 1], ...
    {'channel_crit_excluded','ChannelCriterionExcluded'}, 0.1, ...
    {'channel_crit_maxbad_time','ChannelCriterionMaxBadTime'}, 0.5, ...
    {'burst_crit_refmaxbadchns','BurstCriterionRefMaxBadChns'}, 0.075, ...
    {'burst_crit_reftolerances','BurstCriterionRefTolerances'}, [-5 3.5], ...
    {'burst_flt_settings','BurstFilterSettings'}, [], ...
    {'window_crit_tolerances','WindowCriterionTolerances'},[], ...
    {'flatline_crit','FlatlineCriterion'}, 0.001);

% remove flat-line channels
if ~strcmp(flatline_crit,'off')
    EEG = clean_flatlines(EEG,flatline_crit); end

% high-pass filter the data
if ~strcmp(highpass,'off')
    EEG = clean_drifts(EEG,highpass); end
if nargout > 1
    HP = EEG; end

% remove hopeless channels
if ~strcmp(channel_crit,'off')
    EEG = clean_channels(EEG,channel_crit,channel_crit_excluded,[],channel_crit_maxbad_time); end

% repair bursts
if ~strcmp(burst_crit,'off')
    EEG = repair_bursts(EEG,burst_crit,[],[],[],burst_crit_refmaxbadchns,burst_crit_reftolerances,[],burst_flt_settings); end
if nargout > 2
    BUR = EEG; end

% remove hopeless time windows
if ~strcmp(window_crit,'off')
    disp('Now doing final post-cleanup of the output.');
    EEG = clean_windows(EEG,window_crit,window_crit_tolerances); 
end

end




function res = parse_arguments(args, varargin)
% Helper function: Convert a list of name-value pairs into a struct with values assigned to names.

% parse the defaults
defnames = varargin(1:2:end);
defvalues = varargin(2:2:end);
% make a remapping table for alternative argument names
for k=find(cellfun('isclass',defnames,'cell'))
    for l=2:length(defnames{k})
        name_for_alternative.(defnames{k}{l}) = defnames{k}{1}; end
    defnames{k} = defnames{k}{1};
end
% use only the last assignment for each name
[s,indices] = sort(defnames(:));
indices( strcmp(s((1:end-1)'),s((2:end)'))) = [];
% and make the struct
res = cell2struct(defvalues(indices),defnames(indices),2);
% check if the NVPs are indeed NVPs
if ~isempty(args) && ~ischar(args{1})
    args = [defnames(1:length(args)); args]; end
% remap alternative argument names
if exist('name_for_alternative','var')
    for k=1:2:numel(args)
        if isfield(name_for_alternative,args{k})
            args{k} = name_for_alternative.(args{k}); end
    end
end
% override defaults with arguments to create the output struct
for k=1:2:numel(args)
    if ~isfield(res,args{k})
        error(['Undefined argument name: ' args{k}]); end
    if ~isempty(args{k+1})
        res.(args{k}) = args{k+1}; end
end
% copy to the caller's workspace if no output requested
if nargout == 0
    for fn=fieldnames(res)'
        assignin('caller',fn{1},res.(fn{1})); end
end
end
