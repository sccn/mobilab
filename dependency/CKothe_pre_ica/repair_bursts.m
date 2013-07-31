function signal = repair_bursts(signal,stddev_cutoff,window_len,block_size,max_dimensions,ref_maxbadchannels,ref_tolerances,ref_wndlen,flt_settings)
% Projects low-dimensional burst artifacts out of the data.
% Signal = repair_bursts(Signal,StandardDevCutoff,WindowLength,BlockSize,MaxDimensions,ReferenceMaxBadChannels,RefTolerances,ReferenceWindowLength,SpectralFilterSettings)
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
%   Signal          : continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 1.0Hz transition band)
%
%   StandardDevCutoff: StdDev cutoff for rejection. Data segments whose variance is beyond this cutoff 
%                      from the distribution of variance across the recording are considered missing data.
%                      The reasonable range is 3 (fairly aggressive), 4 (fairly safe), or 5 (very safe,
%                      likely not harming any EEG). Default: 3.
%
%   The following are "specialty' parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these better to your data.
%
%   WindowLength    : Length of the statistcs window, in seconds. This should not be much longer 
%                     than the time scale over which artifacts persist, but the number of samples 
%                     in the window should not be smaller than 1.5x the number of channels. Default:
%                     max(0.5,1.5*Signal.nbchan/Signal.srate);
%
%   BlockSize       : Block granularity for processing. The reprojection matrix will be updated every 
%                     this many samples and a blended matrix is used for the in-between samples. If 
%                     empty this will be set the WindowLength/2 in samples. Default: []
%
%   MaxDimensions   : Maximum dimensionality to reconstruct. Up to this many dimensions (or up to 
%                     this fraction of dimensions) can be reconstructed for a given data segment. This is 
%                     since the lower eigenvalues are usually not estimated very well. Default: 2/3.
%
%   ReferenceMaxBadChannels: The maximum fraction of bad channels per time window of the data that 
%                            is used as clean reference EEG on which statistics are based. Instead
%                            of a number one may also directly pass in a data set that contains
%                            clean reference data (for example a minute of resting EEG). A lower
%                            values makes this criterion more aggressive. Reasonable range: 0.05
%                            (very aggressive) to 0.3 (quite lax). Default: 0.075.
%
%   ReferenceTolerances : These are the power tolerances beyond which a channel in the clean 
%                         reference data is considered "bad", in standard deviations relative
%                         to a robust EEG power distribution (lower and upper bound). 
%                         Default: [-5 3.5].
%
%   ReferenceWindowLength : Length of the windows that are considered reference-quality EEG. 
%                           Default: 1.
%
%   SpectralFilterSettings : Parameters for the spectrum-shaping filter that reweights frequencies
%                            according to their relevance for artifact detection. This is a cell
%                            array of {Frequencies,Amplitudes,Order} for a Yule-Walker IIR filter
%                            design. Note: These settings are very sensitive, do not change unless
%                            you know exactly what you're doing. Can also be the string 'basic'
%                            (for the basic 6th order setting) or 'sensitive' (for a high-frequency
%                            sensitive 8th order setting). Default: 'sensitive'.
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

if ~exist('stddev_cutoff','var') || isempty(stddev_cutoff) stddev_cutoff = 3; end
if ~exist('window_len','var') || isempty(window_len) window_len = max(0.5,1.5*signal.nbchan/signal.srate); end
if ~exist('block_size','var') || isempty(block_size) block_size = []; end
if ~exist('max_dimensions','var') || isempty(max_dimensions) max_dimensions = 0.66; end
if ~exist('ref_maxbadchannels','var') || isempty(ref_maxbadchannels) ref_maxbadchannels = 0.075; end
if ~exist('ref_tolerances','var') || isempty(ref_tolerances) ref_tolerances = [-5 3.5]; end
if ~exist('ref_wndlen','var') || isempty(ref_wndlen) ref_wndlen = 7; end
if ~exist('flt_settings','var') || isempty(flt_settings) flt_settings = 'sensitive'; end

global chanlocs; chanlocs = signal.chanlocs;

% determine the filter setting to use
if strcmp(flt_settings,'sensitive') && signal.srate > 512
    disp('NOTE: To use repair_bursts with the ''sensitive'' setting you need to reduce the sampling rate to at most 512 Hz.');
    flt_settings = 'basic';
end
if strcmp(flt_settings,'basic')
    flt_settings = {[0 2 3 13 14],[1 0.75 0.3 0.3 1],6};
elseif strcmp(flt_settings,'sensitive')
    flt_settings = {[0 2 3 13 16 40 min(80,signal.srate/2-1)],[3 0.75 0.33 0.33 1 1 3],8};
elseif ischar(flt_settings)
    error('Unknown filter setting.');
end

% first find a section of reference data...
disp('Finding a clean section of the data...');
if isnumeric(ref_maxbadchannels)
    ref_section = clean_windows(signal,ref_maxbadchannels,ref_tolerances,ref_wndlen); end

% calibrate on the reference data
disp('Calibrating reference measure...');
[shaping{1:2}] = yulewalk(flt_settings{3},[2*flt_settings{1}/signal.srate 1],flt_settings{2}([1:end end]));
[cov_mat,mix_mat,state_o0,state_o1,state_o2,state_buf] = asr_calibrate(ref_section.data,ref_section.srate,window_len,window_len/2,shaping{:}); clear ref_section;

disp('Now cleaning; this may take a while...');
max_memory = 0.25 * java.lang.management.ManagementFactory.getOperatingSystemMXBean().getFreePhysicalMemorySize()/1024/1024;
if isempty(block_size)
    block_size = floor(signal.srate*window_len/2); end
signal.data = asr_process([signal.data state_buf],signal.srate,window_len,window_len/2,stddev_cutoff,block_size,max_dimensions,max_memory,cov_mat,mix_mat,shaping{:},state_o0,state_o1,state_o2,state_buf);
signal.data(:,1:size(state_buf,2)) = [];
