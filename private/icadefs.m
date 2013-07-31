% icadefs() - function to read in a set of EEGLAB system-wide (i.e. lab-wide)
%             or working directory-wide constants and preferences. Change the 
%             way these are defined in the master icadefs.m file (usually
%             in dir eeglab/functions/sigprocfunc) or make a custom copy of 
%             the icadefs.m file in a project directory. Then, calling functions 
%             that call icadefs from an EEGLAB session in that working directory 
%             will read the local copy, which may set preferences different from 
%             the system-wide copy.
%
% Author: Arnaud Delorme, Scott Makeig, SCCN/INC/UCSD, La Jolla, 05-20-97 

% Copyright (C) 05-20-97 Scott Makeig, SCCN/INC/UCSD, scott@sccn.ucsd.edu
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

% ----------------------------------------------------------------------
% ------ EEGLAB DEFINITION - YOU MAY CHANGE THE TEXT BELOW -------------
% ----------------------------------------------------------------------

% Set EEGLAB figure and GUI colors
% --------------------------------
BACKCOLOR           = [.93 .96 1];    % EEGLAB Background figure color

try
    mobilab = evalin('base','mobilab');
    BACKEEGLABCOLOR = mobilab.preferences.gui.backgroundColor;
    BACKCOLOR = mobilab.preferences.gui.buttonColor;
catch %#ok
    BACKEEGLABCOLOR     = [.66 .76 1];    % EEGLAB main window background
end
    
GUIBUTTONCOLOR      = BACKEEGLABCOLOR;% Buttons colors in figures
GUIPOPBUTTONCOLOR   = BACKCOLOR;      % Buttons colors in GUI windows
GUIBACKCOLOR        = BACKEEGLABCOLOR;% EEGLAB GUI background color <---------
GUITEXTCOLOR        = [0 0 0.4];      % GUI foreground color for text
PLUGINMENUCOLOR     = [.5 0 .5];      % plugin menu color

SC  =  ['binica.sc'];           % Master .sc script file for binica.m
                                % MATLAB will use first such file found
                                % in its path of script directories.
                                % Copy to pwd to alter ICA defaults