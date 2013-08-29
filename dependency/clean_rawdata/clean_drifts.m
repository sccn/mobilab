function signal = clean_drifts(signal,transition)
% Removes drifts from the data using a forward-backward high-pass filter.
% Signal = clean_drifts(Signal,Transition)
%
% This removes drifts from the data using a forward-backward (non-causal) filter.
% NOTE: If you are doing directed information flow analysis, do no use this filter but some other one.
%
% In:
%   Signal : the continuous data to filter
%
%   Transition : the transition band in Hz, i.e. lower and upper edge of the transition
%                (default: [0.5 1])
%
% Out:
%   Signal : the filtered signal
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-09-01

if ~exist('transition','var') || isempty(transition) transition = [0.5 1]; end

% design a FIR highpass filter
N = firpmord(transition, [0 1], [0.001 0.01], signal.srate);
B = fir2(N,[0 transition/(signal.srate/2) 1],[0 0 1 1]);

% apply it
signal.data = signal.data';
for c=1:signal.nbchan
    signal.data(:,c) = filtfilt_fast(B,1,signal.data(:,c)); end
signal.data = signal.data';
signal.etc.clean_drifts_kernel = B;




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
