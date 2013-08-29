function state = asr_calibrate(X,srate,cutoff,blocksize,B,A)
% Calibrate the ASR method.
% State = asr_calibrate(Data,SamplingRate,Cutoff,BlockSize,FilterB,FilterA,UseGPU)
%
% The input to this data is a multi-channel time series of calibration data. The calibration data
% should be reasonably clean resting EEG of ca. 1 minute duration (can also be longer).
%
% The calibration data must have been recorded for the same cap design that shall be used online,
% and ideally should be from the same session and same subject, but it is possible to reuse the
% calibration data from a previous session and montage as long as the cap is placed approx. in the
% same location.
%
% The calibration data should have been high-pass filtered (for example at 0.5Hz or 1Hz using a
% Butterworth IIR filter).
%
% In:
%   Data : Calibration data [#channels x #samples]; high-pass filtered and reasonably clean EEG of 
%          no less than 30 seconds length. The data should be high-pass filtered.
%
%   SamplingRate : Sampling rate of the data, in Hz.
%
%   RejectionCutoff: Standard deviation cutoff for rejection. Data portions whose variance is larger
%                    than this threshold relative to the calibration data is considered missing data
%                    and will be removed. The most aggressive value that can be used without losing 
%                    too much EEG is 1.5. A very conservative value would be 4. Default: 2.5
%
%   Blocksize : Block size for calculating the robust data covariance, in samples; allows to reduce 
%               the memory requirements of the robust estimator by this factor (down to
%               Channels x Channels x Samples x 16 / Blocksize bytes). Default: 10
%
%   FilterB, FilterA : Coefficients of an IIR filter that is used to shape the spectrum of the signal
%                      when calculating artifact statistics. The output signal does not go through this
%                      filter. Default: 
%                      [b,a] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]);
%
% Out:
%   State : initial state struct for asr_process
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-08-31

[C,S] = size(X);

if nargin < 3 || isempty(cutoff)
    cutoff = 2.5; end
if nargin < 4 || isempty(blocksize)
    blocksize = 10; end
blocksize = max(blocksize,ceil((C*C*S*8*3*2)/hlp_memfree));
if nargin < 6 || isempty(A) || isempty(B)
    [B,A] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]); end

X(~isfinite(X(:))) = 0;

% apply the signal shaping filter and initialize the IIR filter state
[X,iirstate] = filter(B,A,double(X),[],2); X = X';
if any(~isfinite(X(:)))
    error('The IIR filter diverged on your data. Please try using either a more conservative filter or removing some bad sections/channels from the calibration data.'); end

% calculate the sample covariance matrices U (averaged in blocks of blocksize successive samples)
U = zeros(length(1:blocksize:S),C*C);
for k=1:blocksize
    range = min(S,k:blocksize:(S+k-1));
    U = U + reshape(bsxfun(@times,reshape(X(range,:),[],1,C),reshape(X(range,:),[],C,1)),size(U));
end

% get the mixing matrix M
M = sqrtm(real(reshape(block_geometric_median(U/blocksize),C,C)));

% get the threshold matrix T
[V,D] = eig(M); %#ok<NASGU>
X = abs(X*V);
T = diag(median(X) + cutoff*1.3652*mad(X,1))*V';

% initialize the remaining filter state
state = struct('M',M,'T',T,'B',B,'A',A,'cov',[],'carry',[],'iir',iirstate);
