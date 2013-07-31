function signal = clean_spikes(signal,qq)
% Set outliers in data to zero.
% Signal = clean_spikes(Signal,Quantile)
%
% Note: this filter is a quick test -- it does not quite cut it.
%
% In:
%   Signal   : a continous data set
%
%   Quantile : quantile of the data that should be retained (data higher than that is set to zero).
%              (default: 0.97)
% 
% Out:
%   Signal   : data set with outliers set to zero
% 
%                                  Laura Frolich, Swartz Center for Computational Neuroscience, UCSD
%                                  2010-09-22


if ~exist('qq','var') || isempty(qq) qq = 0.97; end

for i = size(signal.data,1) %#ok<*NODEF>
    signal.data(i,abs(signal.data(i,:)) > quantile(abs(signal.data(i,:)),qq)) = 0; end

