function signal = repair_bursts(signal,stddev_cutoff,window_len,block_size,max_dimensions,ref_section,ref_channels,ref_wndlen,flt_settings)
% Projects low-dimensional burst artifacts out of the data.
% Signal = repair_bursts(Signal,StandardDevCutoff,WindowLength,BlockSize,MaxDimensions,RefSection,RefChannels)
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
%                      (default: 3)
%
%
%   The following are "specialty' parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these better to your data.
%
%   WindowLength    : Length of the statistcs window, in seconds. This should not be much longer 
%                     than the time scale over which artifacts persist, but the number of samples 
%                     in the window should not be smaller than 1.5x the number of channels (default:
%                     0.5 or as much as required to handle all channels).
%
%   BlockSize       : Block granularity for processing. The reprojection matrix will be updated every 
%                     this many samples and a blended matrix is used for the in-between samples.
%                     (default: 32)
%
%   MaxDimensions   : Maximum dimensionality to reconstruct. Up to this many dimensions (or up to 
%                     this fraction of dimensions) can be reconstructed for a given data segment. This is 
%                     since the lower eigenvalues are usually not estimated very well. (default: 2/3)
%
%   ReferenceSection : This is the quantile of the data that is used as clean reference EEG. The actual
%                      rule is a bit more complicated than just taking a quantile since this is
%                      calculated for each channel separately and then a subset of that is retained
%                      (default: 0.66). Instead of a number one may also directly pass in a data set
%                      that contains clean reference data (for example a minute of resting EEG).
%
%   ReferenceChannels : this is the maximum fraction of channels in the reference section that may
%                       be bad. Default: 0.33. Reasonable range: 0.25 (quite aggressive) to 0.5
%                       (quite lax).
%
%   ReferenceWindowLength : Length of the windows that are considered reference-quality EEG. This 
%                           should be on a time scale of multiple seconds and match the duration
%                           of what can be expected to be a good clean-data period. Default: 10.
%
%   SpectralFilterSettings : Parameters for the spectrum-shaping filter that reweights frequencies
%                            according to their relevance for artifact detection. This is a cell array
%                            of {Frequencies,Amplitudes,Order} for a Yule-Walker IIR filter design. 
%                            Default: {[0 2 3 13 14],[1 0.75 0.3 0.3 1],6}. Note: These settings are
%                            extremely sensitive, do not change unless you know exactly what you're doing.
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
if ~exist('block_size','var') || isempty(block_size) block_size = 32; end
if ~exist('max_dimensions','var') || isempty(max_dimensions) max_dimensions = 0.66; end
if ~exist('ref_section','var') || isempty(ref_section) ref_section = 0.75; end
if ~exist('ref_channels','var') || isempty(ref_channels) ref_channels = 0.33; end
if ~exist('ref_wndlen','var') || isempty(ref_wndlen) ref_wndlen = 7; end
if ~exist('flt_settings','var') || isempty(flt_settings) flt_settings = {[0 2 3 13 14],[1 0.75 0.3 0.3 1],6}; end

% first find a section of reference data...
disp('Finding a clean section of the data...');
if isnumeric(ref_section)
    ref_section = clean_windows(signal,1-ref_section,ref_wndlen,ref_channels); end

% calibrate on the reference data
disp('Calibrating reference measure...');
[shaping{1:2}] = yulewalk(flt_settings{3},[2*flt_settings{1}/signal.srate 1],flt_settings{2}([1:end end]));
[cov_mat,mix_mat,state_o0,state_o1,state_o2,state_buf] = asr_calibrate(ref_section.data,ref_section.srate,window_len,window_len/2,shaping{:}); clear ref_section;

disp('Now cleaning; this may take a while...');
max_memory = 0.25 * java.lang.management.ManagementFactory.getOperatingSystemMXBean().getFreePhysicalMemorySize()/1024/1024;
signal.data = asr_process([signal.data state_buf],signal.srate,window_len,window_len/2,stddev_cutoff,block_size,max_dimensions,max_memory,cov_mat,mix_mat,shaping{:},state_o0,state_o1,state_o2,state_buf);
signal.data(:,1:size(state_buf,2)) = [];
