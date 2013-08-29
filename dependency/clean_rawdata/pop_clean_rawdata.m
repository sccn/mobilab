% pop_clean_rawdata(): Launch GUI to collect user inputs for
%                      clean_artifacts().
% Usage:
%   >>  EEG = pop_clean_rawdata();
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

function EEG = pop_clean_rawdata(EEG)

userInput = inputgui('title', 'pop_clean_rawdata()', 'geom', ...
   {{2 6 [0 0] [1 1]}   {2 6 [1 0] [1 1]} ...
    {2 6 [0 1] [1 1]}   {2 6 [1 1] [1 1]} ...
    {2 6 [0 2] [1 1]}   {2 6 [1 2] [1 1]} ...
    {2 6 [0 3] [1 1]}   {2 6 [1 3] [1 1]} ...
    {2 6 [0 4] [1 1]}   {2 6 [1 4] [1 1]} ...
    {2 6 [0 5] [1 1]}   {2 6 [1 5] [1 1]}}, ... 
'uilist',...
   {{'style' 'text' 'string' 'Remove channel if flat more than [sec|-1->off]'} {'style' 'edit' 'string' '5'    } ...
    {'style' 'text' 'string' 'High-pass filt tran band width [F1 F2|-1->off]'} {'style' 'edit' 'string' '0.5 1'} ...
    {'style' 'text' 'string' 'Remove bad channels (see help) [0-1|-1->off]'}   {'style' 'edit' 'string' '0.45' } ...
    {'style' 'text' 'string' 'Repair bursts (see help) [std|-1->off]'}         {'style' 'edit' 'string' '3'    } ...
    {'style' 'text' 'string' 'Remove time windows (see help) [0-1|''off'']'}   {'style' 'edit' 'string' '0.15' } ...
    {'style' 'text' 'string' 'Show results for comparison? (beta version)'}    {'style' 'popupmenu' 'string' 'Yes|No'}});

if isempty(userInput)
    error('Operation terminated by user.')
end

arg_flatline = str2double(userInput{1,1});
arg_highpass = str2double(userInput{1,2});
arg_channel  = str2double(userInput{1,3});
arg_burst    = str2double(userInput{1,4});
arg_window   = str2double(userInput{1,5});
arg_visartfc = userInput{1,6};

cleanEEG = clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_burst, arg_window);
if arg_visartfc == 1; vis_artifacts(cleanEEG,EEG); end
EEG = cleanEEG;
com = sprintf('EEG = pop_clean_rawdata(EEG, %s, [%s], %s, %s, %s, %s);', userInput{1,1}, userInput{1,2}, userInput{1,3}, userInput{1,4}, userInput{1,5}, num2str(userInput{1,6}));
EEG = eegh(com, EEG);
disp('Done.')
return
