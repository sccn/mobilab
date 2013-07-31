function X = filtfilt_fast(varargin)
% Y = filtfilt_fast(B,A,X)
% Equivalent to filtfilt(), except for being faster when both the filter and the signal are long (and A=1), by using FFT convolution (needs fftfilt).
% The function is faster than filter when approx. length(B)>256 and size(X,Dim)>1024, otherwise slower (due size-testing overhead).
%
% Note:
%  Can also be called with four arguments, as Y = filtfilt_fast(N,F,A,X), 
%  in which case an Nth order FIR filter is designed that has the desired frequency response A at normalized frequencies F;
%  F must be 0<=F<=1, and must be 0 and 1 at its both ends, respectively. The function fir2 is used for frequency-sampling filter design.
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
