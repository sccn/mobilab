% clean_rawdata(): a wrapper to call Christian's clean_artifacts.
% Usage:
%   >>  EEG = clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_burst, arg_window)
%
% This function removes drifts, bad channels, generic short-time bursts and bad segments from the data.
% Tip: Any parameter can also be passed in as [] to use the respective default or as 'off' to disable it.
% 
% ------------------ below is from clean_artifacts -----------------------
%
% Hopefully parameter tuning should be the exception when using this function -- the only parameter
% that likely requires a setting is the BurstCriterion. For a clean ERP experiment with little room
% for subject movement the recommended setting is 4. For movement experiments or otherwise noisy
% recordings the default setting of 3 is okay. See also examples at the bottom of the help.
%
%   FlatlineCriterion: Maximum tolerated flatline duration. In seconds. If a channel has a longer
%                      flatline than this, it will be considered abnormal. Default: 5
%
%   Highpass         : Transition band for the initial high-pass filter in Hz. This is formatted as
%                     [transition-start, transition-end]. Default: [0.5 1].
%
%   ChannelCriterion : Criterion for removing bad channels. This is a minimum correlation
%                      value that a given channel must have w.r.t. a fraction of other channels. A
%                      higher value makes the criterion more aggressive. Reasonable range: 0.4 (very
%                      lax) - 0.6 (quite aggressive). Default: 0.45.
%
%   BurstCriterion   : Criterion for projecting local bursts out of the data. This is in standard
%                      deviations from clean EEG at which a signal component would be considered a
%                      burst artifact. Generally a lower value makes the criterion more aggressive.
%                      Reasonable range: 2.5 (very aggressive, cuts some EEG) to 5 (very lax, cuts
%                      almost never EEG). Default: 3.
%
%   WindowCriterion  : Criterion for removing bad time windows. This is the maximum fraction of bad
%                      channels that are tolerated in the final output data for each considered window.
%                      Generally a lower value makes the criterion more aggressive. Default: 0.05.
%                      Reasonable range: 0.05 (very aggressive) to 0.3 (very lax).
%
%   see also: clean_artifacts

% Author: Makoto Miyakoshi and Christian Kothe, SCCN,INC,UCSD
% History:
% 06/26/2012 ver 1.0 by Makoto. Created.

% Copyright (C) 2013, Makoto Miyakoshi SCCN,INC,UCSD
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function cleanEEG = clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_burst, arg_window)

if arg_flatline == -1; arg_flatline = 'off'; disp('flatchan rej disabled.'  ); end
if arg_highpass == -1; arg_highpass = 'off'; disp('highpass disabled.'      ); end
if arg_channel  == -1; arg_channel  = 'off'; disp('badchan rej disabled.'   ); end
if arg_burst    == -1; arg_burst    = 'off'; disp('burst clean disabled.'   ); end
if arg_window   == -1; arg_window   = 'off'; disp('bad window rej disabled.'); end

cleanEEG = clean_artifacts(EEG, 'FlatlineCriterion', arg_flatline,...
                                'Highpass',          arg_highpass,...
                                'ChannelCriterion',  arg_channel,...
                                'BurstCriterion',    arg_burst,...
                                'WindowCriterion',   arg_window);