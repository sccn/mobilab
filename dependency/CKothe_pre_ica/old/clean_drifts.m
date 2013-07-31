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
