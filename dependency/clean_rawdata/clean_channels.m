function signal = clean_channels(signal,min_corr,ignored_quantile,window_len,max_broken_time,linenoise_aware)
% Remove channels with abnormal data from a continuous data set.
% Signal = clean_channels(Signal,MinCorrelation,IgnoredQuantile,WindowLength,MaxBrokenTime,LineNoiseAware)
%
% This is an automated artifact rejection function which ensures that the data contains no channels
% that record only noise for extended periods of time. If channels with control signals are
% contained in the data these are usually also removed. The criterion is based on correlation: if a
% channel is decorrelated from all others (pairwise correlation < a given threshold), excluding a
% given fraction of most correlated channels -- and if this holds on for a sufficiently long fraction 
% of the data set -- then the channel is removed.
%
% In:
%   Signal          : Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 2.0Hz transition band).
%
%   MinCorrelation  : Minimum correlation between a channel and any other channel (in a short period 
%                     of time) below which the channel is considered abnormal for that time period.
%                     Reasonable range: 0.4 (very lax) to 0.6 (quite aggressive). The default is 0.45. 
%                     
%
%   The following are "specialty" parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these to your data.
%   
%   IgnoredQuantile : Fraction of channels that need to have at least the given MinCorrelation value
%                     w.r.t. the channel under consideration. This allows to deal with channels or
%                     small groups of channels that measure the same noise source, e.g. if they are
%                     shorted. If many channels can be disconnected during an experiment and you
%                     have strong noise in the room, you might increase this fraction, but consider
%                     that this a) requires you to descrease the MinCorrelation appropriately and b)
%                     can make the correlation measure more brittle. Reasonable range: 0.05 (rather
%                     lax) to 0.2 (very tolerant re disconnected/shorted channels).The default is
%                     0.1.
%
%   WindowLength    : Length of the windows (in seconds) for which correlation is computed; ideally
%                     short enough to reasonably capture periods of global artifacts (which are
%                     ignored), but not shorter (for computational reasons). Default: 2.
% 
%   MaxBrokenTime : Maximum time (either in seconds or as fraction of the recording) during which a 
%                   retained channel may be broken. Reasonable range: 0.1 (very aggressive) to 0.6
%                   (very lax). The default is 0.5.
%
%   LineNoiseAware : Whether the operation should be performed in a line-noise aware manner. If enabled,
%                    the correlation measure will not be affected by the presence or absence of line 
%                    noise. Default: true.
%
% Out:
%   Signal : data set with bad channels removed
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

if ~exist('min_corr','var') || isempty(min_corr) min_corr = 0.45; end
if ~exist('ignored_quantile','var') || isempty(ignored_quantile) ignored_quantile = 0.1; end
if ~exist('window_len','var') || isempty(window_len) window_len = 2; end
if ~exist('max_broken_time','var') || isempty(max_broken_time) max_broken_time = 0.5; end
if ~exist('linenoise_aware','var') || isempty(linenoise_aware) linenoise_aware = true; end

% flag channels
if ~exist('removed_channels','var')
    if max_broken_time > 0 && max_broken_time < 1  %#ok<*NODEF>
        max_broken_time = size(signal.data,2)*max_broken_time;
    else
        max_broken_time = signal.srate*max_broken_time;
    end
    
    [C,S] = size(signal.data);
    window_len = window_len*signal.srate;
    wnd = 0:window_len-1;
    offsets = 1:window_len:S-window_len;
    W = length(offsets);    
    retained = 1:(C-ceil(C*ignored_quantile));
        
    % optionally ignore both 50 and 60 Hz spectral components...
    if linenoise_aware
        if signal.srate <= 130
            B = fir2(500,[2*[0 45 50 55]/signal.srate 1],[1 1 0 1 1]);
        else
            B = fir2(500,[2*[0 45 50 55 60 65]/signal.srate 1],[1 1 0 1 0 1 1]);
        end
        X = filtfilt_fast(B,1,signal.data');
    else
        X = signal.data';
    end

    flagged = zeros(C,W);
    % for each window, flag channels with too low correlation to any other channel (outside the
    % ignored quantile)
    for o=1:W
        sortcc = sort(abs(corrcoef(X(offsets(o)+wnd,:))));
        flagged(:,o) = all(sortcc(retained,:) < min_corr);
    end
    % mark all channels for removal which have more flagged samples than the maximum number of
    % ignored samples
    removed_channels = sum(flagged,2)*window_len > max_broken_time;
end

% execute
if any(removed_channels)
    disp('Now removing bad channels...');
    signal = pop_select(signal,'nochannel',find(removed_channels));
    if isfield(signal.etc,'clean_channel_mask')
        signal.etc.clean_channel_mask(signal.etc.clean_channel_mask) = ~removed_channels;
    else
        signal.etc.clean_channel_mask = ~removed_channels;
    end
end



function X = filtfilt_fast(varargin)
% Like filtfilt(), but faster when filter and signal are long (and A=1).
% Y = filtfilt_fast(B,A,X)
%
% Uses FFT convolution (needs fftfilt). The function is faster than filter when approx. length(B)>256 and size(X,Dim)>1024, 
% otherwise slower (due size-testing overhead).
%
% Note:
%  Can also be called with four arguments, as Y = filtfilt_fast(N,F,A,X), 
%  in which case an Nth order FIR filter is designed that has the desired frequency response A at normalized frequencies F;
%  F must be 0<=F<=1, and must be 0 and 1 at its both ends, respectively. The function fir2 is used for frequency-sampling filter design.
%
% See also: 
%   filtfilt, filter
% 
%                           Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                           2010-07-14

if nargin == 3
    [B A X] = deal(varargin{:});
elseif nargin == 4
    [N F M X] = deal(varargin{:});
    B = fir2(N,F,sqrt(M)); A = 1;
else
    help filtfilt_fast;
    return;
end

if A == 1
    was_single = strcmp(class(X),'single');
    w = length(B); t = size(X,1);    
    % extrapolate
    X = double([bsxfun(@minus,2*X(1,:),X((w+1):-1:2,:)); X; bsxfun(@minus,2*X(t,:),X((t-1):-1:t-w,:))]);
    % filter, reverse
    X = filter_fast(B,A,X); X = X(length(X):-1:1,:);
    % filter, reverse
    X = filter_fast(B,A,X); X = X(length(X):-1:1,:);
    % remove extrapolated pieces
    X([1:w t+w+(1:w)],:) = [];
    if was_single
        X = single(X); end    
else    
    % fall back to filtfilt for the IIR case
    X = filtfilt(B,A,X);
end



function [X,Zf] = filter_fast(B,A,X,Zi,dim)
% Like filter(), but faster when both the filter and the signal are long.
% [Y,Zf] = filter_fast(B,A,X,Zi,Dim)
%
% Uses FFT convolution (needs fftfilt). The function is faster than filter when approx. length(B)>256 and size(X,Dim)>1024,
% otherwise slower (due size-testing overhead).
%
% See also:
%   filter, fftfilt
%
%                           Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                           2010-07-09

persistent has_fftfilt;
if isempty(has_fftfilt)
    has_fftfilt = exist('fftfilt','file');
    % see if we also have the license...
    try
        x=fftfilt();
    catch e
        if strcmp(e.identifier,'MATLAB:UndefinedFunction')
            has_fftfilt = false; end
    end
end

if nargin <= 4
    dim = find(size(X)~=1,1); end
if nargin <= 3
    Zi = []; end

lenx = size(X,dim);
lenb = length(B);
if lenx == 0
    % empty X
    Zf = Zi;
elseif lenb < 256 || lenx<1024 || lenx <= lenb || lenx*lenb < 4000000 || ~isequal(A,1) || ~has_fftfilt
    % use the regular filter
    if nargout > 1
        [X,Zf] = filter(B,A,X,Zi,dim);
    else
        X = filter(B,A,X,Zi,dim);
    end
else
    was_single = strcmp(class(X),'single');
    % fftfilt can be used
    if isempty(Zi)
        % no initial conditions to take care of
        if nargout < 2
            % and no final ones
            X = unflip(fftfilt(B,flip(double(X),dim)),dim);
        else
            % final conditions needed
            X = flip(X,dim);
            [dummy,Zf] = filter(B,1,X(end-length(B)+1:end,:),Zi,1); %#ok<ASGLU>
            X = fftfilt(B,double(X));
            X = unflip(X,dim);
        end
    else
        % initial conditions available
        X = flip(X,dim);
        % get a Zi-informed piece
        tmp = filter(B,1,X(1:length(B),:),Zi,1);
        if nargout > 1
            % also need final conditions
            [dummy,Zf] = filter(B,1,X(end-length(B)+1:end,:),Zi,1); %#ok<ASGLU>
        end
        X = fftfilt(B,double(X));
        % incorporate the piece
        X(1:length(B),:) = tmp;
        X = unflip(X,dim);
    end
    if was_single
        X = single(X); end
end

function X = flip(X,dim)
if dim ~= 1
    order = 1:ndims(X);
    order = order([dim 1]);
    X = permute(X,order);
end

function X = unflip(X,dim)
if dim ~= 1
    order = 1:ndims(X);
    order = order([dim 1]);
    X = ipermute(X,order);
end
