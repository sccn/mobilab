% eegplugin_clean_rawdata() - a wrapper to plug-in Christian's clean_artifact() into EEGLAB. .
% 
% Usage:
%   >> eegplugin_firstPassOutlierTrimmer(fig,try_strings,catch_strings);
%
%   see also: clean_artifacts

% Author: Makoto Miyakoshi and Christian Kothe, SCCN,INC,UCSD 2013
% History:
% 06/26/2013 ver 1.0 by Makoto. Created.

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

function eegplugin_clean_rawdata(fig,try_strings,catch_strings)

% create menu
toolsmenu = findobj(fig, 'tag', 'tools');
uimenu( toolsmenu, 'label', 'clean_rawdata', 'separator','on',...
    'callback', 'EEG = pop_clean_rawdata(EEG); [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); eeglab redraw');