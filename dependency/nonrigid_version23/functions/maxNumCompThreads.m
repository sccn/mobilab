function N = maxNumCompThreads(varargin)
% maxNumCompThreads returns the number of available CPU cores, works with
% Windows, Linux, OpenBSD and MAC-OS, using a c-coded mex-file.
%
%   N = maxNumCompThreads()
%
% Replacement of the original "Matlab maxNumCompThreads" which will be removed
% in a future release.

N=feature('Numcores');

