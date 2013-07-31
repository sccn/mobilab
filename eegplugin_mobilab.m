% eegplugin_mobi2eeglab() - MOBILAB plugin for EEGLAB
%                           MOBILAB is a collection of functionalities
%                           for processing MoBI data, see (Makeig et al.,
%                           2009).
% Usage:
%   >> eegplugin_mobilab(fig, trystrs, catchstrs);
%
% Inputs:
%   fig        - [integer] eeglab figure.
%   trystrs    - [struct] "try" strings for menu callbacks.
%   catchstrs  - [struct] "catch" strings for menu callbacks.
%
% Author: Alejandro Ojeda, Nima Bigdely Shamlo, and Christian Kothe, SCCN, INC, UCSD
%
% See also: eeglab()

function eegplugin_mobilab(fig,try_strings, catch_strings)

p = fileparts(which('mobilab'));
if isempty(p)
    p = fileparts(which('eeglab'));
    p = [p filesep 'plugins' filesep 'mobilab'];
end
% addpath(genpath([p filesep 'dependency']));
addpath([p filesep 'dependency' filesep 'Hed']);
addpath(genpath([p filesep 'eeglabInterface']));
addpath(genpath([p filesep 'glibmobi']));
addpath(genpath([p filesep 'gui']));

h = findobj(fig,'Label','Import data');

% uimenu( h, 'label', 'From file (.xdf, .xdfz, .drf)', 'callback','allDataStreams = pop_load_file_mobilab;');
% uimenu( h, 'label', 'From folder (concat LSL or DataRiver files)', 'callback','allDataStreams = pop_load_folder_mobilab;');
% uimenu( h, 'label', 'From legacy DataRiver .bdf file', 'callback','allDataStreams = pop_load_bdf;');
% uimenu( h, 'label', 'From MoBILAB folder', 'callback','allDataStreams = pop_load_MoBI;');
uimenu( h, 'label', 'Into MoBILAB', 'callback','runmobilab;');

h = findobj(fig,'Label','Tools');
uimenu( h, 'label', 'MoBILAB','CallBack','disp(''runmobilab'');runmobilab;');