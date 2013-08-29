function signal = repair_bursts(signal,cutoff,windowlen,stepsize,maxdims,ref_maxbadchannels,ref_tolerances,ref_wndlen,usegpu)
% Projects low-dimensional burst artifacts out of the data.
% Signal = repair_bursts(Signal,StandardDevCutoff,WindowLength,BlockSize,MaxDimensions,ReferenceMaxBadChannels,RefTolerances,ReferenceWindowLength)
%
% This is an automated artifact rejection function that ensures that the data contains no events
% that have abnormally strong power; the subspaces on which those events occur are reconstructed 
% (interpolated) based on the rest of the EEG signal during these time periods.
%
% The basic principle is to first find a section of data that represents clean "reference" EEG and
% to compute statistics on there. Then, the function goes over the whole data in a sliding window
% and finds the subspaces in which there is activity that is more than a few standard deviations
% away from the reference EEG (this is a tunable parameter). Once the function has found the bad
% subspaces it will treat them as missing data and reconstruct their content using a mixing matrix
% that was calculated on the clean data.
%
% In:
%   Signal : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or with a 
%            0.5Hz - 1.0Hz transition band)
%
%   Cutoff : StdDev cutoff for rejection. Data segments whose variance is beyond this cutoff from the 
%            distribution of variance across the recording are considered missing data. The
%            reasonable range is 1.5 (fairly aggressive), 4 (fairly safe), or 5 (very safe, likely not
%            harming any EEG). Default: 2.5.
%
%   The following are "specialty' parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these better to your data.
%
%   WindowLength : Length of the statistcs window, in seconds. This should not be much longer 
%                  than the time scale over which artifacts persist, but the number of samples in
%                  the window should not be smaller than 1.5x the number of channels. Default:
%                  max(0.5,1.5*Signal.nbchan/Signal.srate);
%
%   StepSize : Step size for processing. The reprojection matrix will be updated every this many
%              samples and a blended matrix is used for the in-between samples. If empty this will
%              be set the WindowLength/2 in samples. Default: []
%
%   MaxDimensions : Maximum dimensionality to reconstruct. Up to this many dimensions (or up to this 
%                   fraction of dimensions) can be reconstructed for a given data segment. This is
%                   since the lower eigenvalues are usually not estimated very well. Default: 2/3.
%
%   ReferenceMaxBadChannels : The maximum fraction of bad channels per time window of the data that 
%                             is used as clean reference EEG on which statistics are based. Instead
%                             of a number one may also directly pass in a data set that contains
%                             clean reference data (for example a minute of resting EEG). A lower
%                             values makes this criterion more aggressive. Reasonable range: 0.05
%                             (very aggressive) to 0.3 (quite lax). Default: 0.075.
%
%   ReferenceTolerances : These are the power tolerances beyond which a channel in the clean 
%                         reference data is considered "bad", in standard deviations relative
%                         to a robust EEG power distribution (lower and upper bound). 
%                         Default: [-5 3.5].
%
%   ReferenceWindowLength : Length of the windows that are considered reference-quality EEG. 
%                           Default: 1.
%
%   UseGPU : Whether to run on the GPU. Makes sense for offline processing if you have a GTX 780 or 
%            GTX Titan. Default: false
%
% Out:
%   Signal : data set with local peaks removed
%
% Examples:
%   % use the defaults
%   eeg = repair_bursts(eeg);
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-10

if ~exist('cutoff','var') || isempty(cutoff) cutoff = 2.5; end
if ~exist('windowlen','var') || isempty(windowlen) windowlen = max(0.5,1.5*signal.nbchan/signal.srate); end
if ~exist('stepsize','var') || isempty(stepsize) stepsize = []; end
if ~exist('maxdims','var') || isempty(maxdims) maxdims = 0.66; end
if ~exist('ref_maxbadchannels','var') || isempty(ref_maxbadchannels) ref_maxbadchannels = 0.075; end
if ~exist('ref_tolerances','var') || isempty(ref_tolerances) ref_tolerances = [-5 3]; end
if ~exist('ref_wndlen','var') || isempty(ref_wndlen) ref_wndlen = 7; end
if ~exist('usegpu','var') || isempty(usegpu) usegpu = false; end

global chanlocs; chanlocs = signal.chanlocs;

% first find a section of reference data...
%***JRI*** modified to allow documented ability to pass in an EEG structure of clean data ***
if isnumeric(ref_maxbadchannels)
    disp('Finding a clean section of the data...');
    ref_section = clean_windows(signal,ref_maxbadchannels,ref_tolerances,ref_wndlen); 
else
    disp('Using user-supplied clean section of data.')
    ref_section = ref_maxbadchannels; 
end

% calibrate on the reference data
disp('Estimating thresholds; this may take a while...');
state = asr_calibrate(ref_section.data,ref_section.srate,cutoff);
if isempty(stepsize)
    stepsize = floor(signal.srate*windowlen/2); end
[signal.data,state] = asr_process(signal.data,signal.srate,state,windowlen,windowlen/2,stepsize,maxdims,[],usegpu);
signal.data = [signal.data(:,size(state.carry,2)+1:end) state.carry];
%signal.data(:,1:size(state.carry,2)) = [];
